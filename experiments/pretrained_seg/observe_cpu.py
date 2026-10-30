"""CPU evaluation of logits already stitched by pretrained_seg's MONAI inference."""

import time

import nibabel as nib
import numpy as np
import torch
from monai.transforms import SaveImage

from experiments.munet.evaluation import _evaluate_case
from experiments.munet.observe_cpu import feature_records, prediction_record, restored_prediction, selected


def process_case(job):
    """Compute statistics, restore labels, and evaluate one case on the CPU."""
    started = time.perf_counter()
    groups, case = set(job["groups"]), job["case"]
    records = feature_records(job["features"], job["samples"], job["gamma"], job["patch_count"])
    metric_rows = {}
    normal = None
    probability_needed = bool(groups & {"confidence", "distribution"})
    for condition, results in job["outputs"]:
        condition_started = time.perf_counter()
        logits = results[0].float()  # Already stitched by MONAI on the inference device.
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
                        "stitch_seconds": 0.0, "prediction_metrics_seconds": metrics_seconds,
                        "inversion_and_prediction_save_seconds": save_seconds,
                        "quality_and_orthogonal_save_seconds": quality_seconds,
                        "seconds": time.perf_counter() - condition_started})
        # Release the transport list after the case has been processed.
        results.clear()
        del logits, probability, labels
    records.append({"event": "cpu_case_runtime", "case": case,
                    "seconds": time.perf_counter() - started})
    return {"records": records, "metrics": metric_rows}
