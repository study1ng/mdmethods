"""CPU-only observation jobs. Never pass a model or CUDA tensor to this pool."""

import os
import signal
import time

import nibabel as nib
from monai.data import MetaTensor, decollate_batch
from monai.transforms import SaveImage
import numpy as np
import torch

from experiments.callbacks import invert
from experiments.config import image_key
from experiments.munet.evaluation import _evaluate_case, write_metric_rows as write_evaluation_rows
from experiments.munet.model import MUNetTrainingModule
from experiments.munet.stitch_utils import BlendMode, stitch_logits


METRIC_GROUPS = ("basic", "quality", "confidence", "distribution",
                 "bottleneck", "internal", "full-stats")
BOTTLENECK_STAGES = {"encoder_features", "mamba_output", "after_residual"}


def initialize_worker():
    # Spawn avoids inheriting CUDA state. Each process has one CPU thread.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def run_disk_job(function, path):
    """Only a function reference and a path travel through multiprocessing IPC.

    Load ordinary CPU storage from our own temporary file; torch.load without
    mmap does not allocate POSIX shared memory. Results contain scalars/lists,
    never tensors, so the return path does not need tensor reducers either.
    """
    try:
        started = time.perf_counter()
        arguments = torch.load(path, map_location="cpu", weights_only=False)
        read_seconds = time.perf_counter() - started
        result = function(*arguments)
        result["disk_read_seconds"] = read_seconds
        return result
    finally:
        path.unlink(missing_ok=True)


def selected(mode, condition):
    return mode == "all" or (mode == "normal" and condition == "normal")


def comparison(current, reference):
    if current.shape != reference.shape:
        return {"status": "shape_mismatch"}
    a, b = current.double().reshape(-1), reference.double().reshape(-1)
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return {"status": "nonfinite"}
    delta = a - b
    an, bn = a.norm().item(), b.norm().item()
    return {"status": "ok", "elements": a.numel(), "mae": delta.abs().mean().item(),
            "rmse": delta.square().mean().sqrt().item(), "max_abs": delta.abs().max().item(),
            "relative_l2": delta.norm().item() / bn if bn else None,
            "norm_ratio": an / bn if bn else None,
            "cosine": torch.dot(a, b).item() / (an * bn) if an and bn else None}


def full_feature_stats(record, current):
    started = time.perf_counter()
    record = dict(record)
    record.update(MUNetTrainingModule._diagnostic_stats(current))
    if current.ndim >= 3:
        channels = current.reshape(current.shape[0], current.shape[1], -1)
        record["channel_mean"] = channels.mean((0, 2)).tolist()
        record["channel_rms"] = channels.double().square().mean((0, 2)).sqrt().tolist()
    record["feature_l2"] = current.double().norm().item()
    record["cpu_seconds"] = time.perf_counter() - started
    return {"records": [record]}


def feature_records(features, samples, gamma, patch_count):
    references, first, inputs = {}, {}, {}
    records = []
    for feature in features:
        record = dict(feature)
        offset, count = record.pop("sample_slice")
        sample = samples[offset:offset + count]
        condition, stage = record["condition"], record["stage"]
        position = tuple(record["position"])
        record["sample_stats"] = MUNetTrainingModule._diagnostic_stats(sample)
        if torch.isfinite(sample).all():
            record["sample_quantiles"] = dict(zip(
                ("p01", "p05", "p50", "p95", "p99"),
                torch.quantile(sample, torch.tensor([.01, .05, .5, .95, .99])).tolist()))
        key = stage, position
        if condition == "normal":
            references[key] = sample
            record["same_patch_vs_normal"] = {"status": "reference"}
        elif key in references:
            record["same_patch_vs_normal"] = comparison(sample, references[key])
        stage_key = condition, stage
        if stage_key in first:
            baseline, first_position = first[stage_key]
            record["vs_first_patch"] = {**comparison(sample, baseline),
                                        "reference_position": first_position}
        else:
            first[stage_key] = sample, record["position"]
            record["vs_first_patch"] = {"status": "reference", "single_patch": patch_count == 1}
        input_key = condition, position
        if stage == "encoder_features":
            inputs[input_key] = sample
        elif stage == "mamba_output" and input_key in inputs:
            scaled, residual = sample * gamma, inputs[input_key]
            record["scaled_update_vs_encoder"] = comparison(scaled, residual)
            denominator = residual.double().norm().item()
            record["sample_scaled_update_l2_over_encoder_l2"] = (
                scaled.double().norm().item() / denominator if denominator else None)
        elif stage == "after_residual" and input_key in inputs:
            record["output_vs_encoder"] = comparison(sample, inputs.pop(input_key))
        records.append(record)
    return records


def identity_difference(value):
    # Called only after validating the whole reference tensor is finite.
    mean_abs = value.abs().mean(dtype=torch.float64).item()
    return {"status": "ok", "max_abs_diff": 0.0, "mean_abs_diff": 0.0,
            "exact_equal": True, "relative_mean_abs_diff": 0.0 if mean_abs else None,
            "relative_status": "ok" if mean_abs else "zero_reference",
            "reference_mean_abs": mean_abs, "tolerance_rtol": 1e-5,
            "matching_fraction_by_atol": {f"{atol:g}": 1.0 for atol in (0., 1e-8, 1e-6, 1e-4)}}


def prediction_record(case, condition, logits, probability, normal, groups, spacing):
    normal_logits, normal_prob, normal_labels = normal
    labels = logits.argmax(1, keepdim=True)
    record = {"event": "prediction", "case": case, "condition": condition,
              "grid_spacing_plan_units": spacing}
    if "basic" in groups:
        record["changed_fraction"] = (labels != normal_labels).double().mean().item()
    if "confidence" in groups:
        record.update({
            "entropy_mean_nats": -(probability * probability.clamp_min(1e-30).log()).sum(1).mean().item(),
            "confidence_mean": probability.max(1).values.mean().item(),
            "top_two_margin_mean": probability.topk(2, dim=1).values.diff(dim=1).abs().mean().item(),
        })
    if "distribution" in groups:
        record["logits_vs_normal"] = (identity_difference(logits) if condition == "normal" else
                                      MUNetTrainingModule._diagnostic_difference(logits, normal_logits))
        record["probabilities_vs_normal"] = (identity_difference(probability) if condition == "normal" else
                                             MUNetTrainingModule._diagnostic_difference(probability, normal_prob))
        kl = js = 0.0
        if condition != "normal":
            pflat, qflat = probability.flatten(2), normal_prob.flatten(2)
            for start in range(0, pflat.shape[-1], 32768):
                p = pflat[:, :, start:start + 32768].double().clamp_min(1e-30)
                q = qflat[:, :, start:start + 32768].double().clamp_min(1e-30)
                middle = (p + q) * .5
                kl += (p * (p.log() - q.log())).sum().item()
                js += (.5 * (p * (p.log() - middle.log()) + q * (q.log() - middle.log()))).sum().item()
        record["kl_condition_to_normal_nats"] = kl / labels.numel()
        record["js_to_normal_nats"] = js / labels.numel()
    if "basic" in groups or "confidence" in groups:
        classes = []
        for organ in range(logits.shape[1]):
            item = {"class": organ}
            if "basic" in groups:
                predicted, baseline = labels == organ, normal_labels == organ
                count, base_count = predicted.sum().item(), baseline.sum().item()
                intersection = (predicted & baseline).sum().item()
                item.update({"voxels": count, "fraction": count / labels.numel(),
                             "vs_normal_dice": 2 * intersection / (count + base_count) if count + base_count else None,
                             "gained_voxels": count - intersection, "lost_voxels": base_count - intersection})
            if "confidence" in groups:
                item["probability_mean"] = probability[:, organ].mean().item()
            classes.append(item)
        record["classes"] = classes
    return record, labels


def restored_prediction(labels, batch):
    # Same decollation and label inversion as LogCallback, without a disk round trip.
    items, values = decollate_batch(batch), decollate_batch(labels)
    if len(items) != 1 or len(values) != 1:
        raise ValueError("Observation requires one case per inference batch")
    reference, value = items[0][image_key], values[0]
    value = MetaTensor(value, meta=dict(reference.meta))
    return invert(value, reference, is_label=True)


def process_case(job):
    """Process all conditions in one case; cases run concurrently in the pool.

    Keeping a case together avoids copying/recomputing the normal baseline in
    every worker and keeps transform inversion and reference labels off the GPU.
    """
    started = time.perf_counter()
    groups, case = set(job["groups"]), job["case"]
    records = feature_records(job["features"], job["samples"], job["gamma"], job["patch_count"])
    metric_rows = {}
    normal = None
    probability_needed = bool(groups & {"confidence", "distribution"})
    for condition, results in job["outputs"]:
        condition_started = time.perf_counter()
        stitch_started = time.perf_counter()
        logits = stitch_logits(results, BlendMode.GAUSSIAN, output_size=job["output_size"]).float()
        stitch_seconds = time.perf_counter() - stitch_started
        if not torch.isfinite(logits).all():
            raise ValueError(f"Nonfinite logits: {case}, {condition}")
        metrics_started = time.perf_counter()
        probability = logits.softmax(1) if probability_needed else None
        if condition == "normal":
            normal = (logits if "distribution" in groups else None,
                      probability if "distribution" in groups else None,
                      logits.argmax(1, keepdim=True))
        record, labels = prediction_record(case, condition, logits, probability, normal, groups, job["spacing"])
        if groups & {"basic", "confidence", "distribution"}:
            records.append(record)
        metrics_seconds = time.perf_counter() - metrics_started
        save_prediction = selected(job["save_predictions"], condition)
        save_orthogonal = selected(job["save_orthogonal"], condition)
        evaluate = job["dice"] or job["hd"] is not None
        save_seconds = quality_seconds = 0.0
        if save_prediction or save_orthogonal or evaluate:
            save_started = time.perf_counter()
            restored = restored_prediction(labels, job["batch"])
            directory = job["directory"] / condition
            filename = f"test_0_{case}_out.nii.gz"
            if save_prediction:
                directory.mkdir(parents=True, exist_ok=True)
                restored.meta["filename_or_obj"] = filename
                SaveImage(output_dir=directory, output_postfix="", separate_folder=False)(restored)
            save_seconds = time.perf_counter() - save_started
            if evaluate or save_orthogonal:
                quality_started = time.perf_counter()
                orthogonal_dir = directory / "orthogonal" if save_orthogonal else None
                if orthogonal_dir is not None:
                    orthogonal_dir.mkdir(parents=True, exist_ok=True)
                image = nib.Nifti1Image(restored.as_tensor()[0].numpy().astype(np.int64),
                                       restored.affine.numpy(), dtype=np.int64)
                image.header.set_xyzt_units("mm")
                # Match the affine precision of a saved/reloaded NIfTI header,
                # even when prediction files themselves are disabled.
                image = nib.Nifti1Image(image.dataobj, image.header.get_best_affine(),
                                       header=image.header)
                rows = _evaluate_case(
                    case, job["source"], image, job["num_classes"],
                    range(1, job["num_classes"]), job["dice"], job["hd"], orthogonal_dir,
                    prediction_name=filename, save_shared=condition == "normal",
                )
                if evaluate:
                    metric_rows[condition] = rows
                quality_seconds = time.perf_counter() - quality_started
        records.append({"event": "cpu_runtime", "case": case, "condition": condition,
                        "stitch_seconds": stitch_seconds, "prediction_metrics_seconds": metrics_seconds,
                        "inversion_and_prediction_save_seconds": save_seconds,
                        "quality_and_orthogonal_save_seconds": quality_seconds,
                        "seconds": time.perf_counter() - condition_started})
        # Drop each condition's patch logits promptly; normal stitched references
        # remain alive until all paired comparisons are complete.
        results.clear()
        del logits, probability, labels
    records.append({"event": "cpu_case_runtime", "case": case,
                    "seconds": time.perf_counter() - started})
    return {"records": records, "metrics": metric_rows}


def write_metric_rows(directory, cases, by_condition, dice, hd):
    """Reuse val's CSV layout and averaging without rereading predictions."""
    columns = (["dice"] if dice else []) + ([f"hd{hd:g}_mm"] if hd is not None else [])
    for condition, case_rows in by_condition.items():
        rows = [row for name in cases for row in case_rows[name]]
        write_evaluation_rows(rows, directory / condition / "metrics", columns)
