"""Persistent checkpoint observation; GPU work lives in disposable subprocesses."""

import argparse
import csv
import ctypes
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
import uuid
from copy import copy
from datetime import datetime, timezone

import torch
from lightning import Trainer

from experiments import ArgumentAdaptor
from experiments.callbacks import LogCallback
from experiments.config import image_key, label_key
from experiments.munet.evaluation import paired_cases, write_metrics
from experiments.munet.stitch_utils import stitch_logits, BlendMode
from experiments.plan import Plan
from experiments.utils import resolved_path


SCHEMA = 1
CONDITIONS = ("normal", "gamma_zero", "no_ppe", "no_mixing",
              "no_ppe_no_mixing", "reverse_order")


def _positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _parser(custom_class):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=resolved_path)
    parser.add_argument("trained", type=resolved_path)
    parser.add_argument("plan", type=resolved_path)
    parser.add_argument("output", type=resolved_path)
    parser.add_argument("-mh", "--module-help", action="help")
    parser.add_argument("-i", "--interval", type=_positive, default=60)
    parser.add_argument("-w", "--workers", type=_positive_int, default=4)
    parser.add_argument("--dice", action="store_true")
    parser.add_argument("--hd", type=custom_class._hd_percentile, default=None)
    parser.add_argument("-d", "--devices", type=int, nargs="+", default=None,
                        help="Candidate logical CUDA indices; default: all visible GPUs.")
    parser.add_argument("--no-ppe", action="store_true", help="Disable PPE in the baseline.")
    parser.add_argument("--no-mixing", action="store_true", help="Disable mixing in the baseline.")
    parser.add_argument("--feature-samples", type=_positive_int, default=4096,
                        help="Maximum evenly spaced feature elements retained per patch/stage.")
    parser.add_argument("--_checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_epoch", type=int, help=argparse.SUPPRESS)
    return parser


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item) for item in value]
    return value


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def _read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _signature(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _settings(args, meta):
    cases = paired_cases(args.data / image_key, args.data / label_key)
    return {"schema": SCHEMA, "data": str(args.data), "trained": str(args.trained),
            "plan_sha256": _sha256(args.plan), "experiment": meta.experiment_name,
            "dice": args.dice, "hd": args.hd, "feature_samples": args.feature_samples,
            "no_ppe": args.no_ppe, "no_mixing": args.no_mixing,
            "conditions": list(CONDITIONS),
            "cases": {name: {key: _signature(path) for key, path in case.items()}
                      for name, case in cases.items()}}


def _smi(*arguments):
    result = subprocess.run(["nvidia-smi", *arguments], capture_output=True,
                            text=True, check=True, timeout=15)
    return list(csv.reader(result.stdout.strip().splitlines(), skipinitialspace=True))


def _visible_gpu_uuids():
    """Resolve CUDA ordinals through the driver without retaining any context.

    nvidia-smi indices need not equal CUDA indices (FASTEST_FIRST, visibility
    masks, containers). Device enumeration and UUID queries create no context.
    """
    try:
        driver = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
        if driver.cuInit(0) != 0:
            return None
        count = ctypes.c_int()
        if driver.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return None
        get_uuid = getattr(driver, "cuDeviceGetUuid_v2", None) or driver.cuDeviceGetUuid
        identifiers = []
        for index in range(count.value):
            device = ctypes.c_int()
            raw = (ctypes.c_ubyte * 16)()
            if driver.cuDeviceGet(ctypes.byref(device), index) != 0 or get_uuid(ctypes.byref(raw), device) != 0:
                return None
            identifiers.append("GPU-" + str(uuid.UUID(bytes=bytes(raw))))
        return identifiers
    except (OSError, AttributeError):
        return None


def _idle_gpu(candidates):
    """Fail closed on missing/unsupported telemetry. Never create a CUDA context."""
    rows = _smi("--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits")
    processes = _smi("--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits")
    busy = {row[0].strip() for row in processes if len(row) >= 2}
    by_uuid = {row[1].strip(): row for row in rows if len(row) == 4}
    identifiers = _visible_gpu_uuids()
    if not identifiers:
        return None
    if candidates is not None and any(index < 0 or index >= len(identifiers) for index in candidates):
        raise ValueError(f"--devices must index the visible GPUs (count={len(identifiers)})")
    available = []
    for logical, identifier in enumerate(identifiers):
        if candidates is not None and logical not in candidates:
            continue
        row = by_uuid.get(identifier)
        if row is None or identifier in busy:
            continue
        try:
            used, utilization = int(row[2]), int(row[3])
        except ValueError:
            continue
        # Permit only a small driver/display allocation, and zero GPU utilization.
        if 0 <= used <= 64 and utilization == 0:
            available.append((used, logical, identifier, row[0].strip()))
    if not available:
        return None
    used, logical, identifier, physical = min(available)
    return {"uuid": identifier, "logical_index": logical, "nvidia_smi_index": physical,
            "memory_used_mib": used, "utilization_percent": 0}


def _checkpoints(root):
    """Only immutable numbered periodic snapshots, never the mutable last.ckpt."""
    found = {}
    for path in root.rglob("*"):
        if not path.is_file() or not path.name.endswith((".ckpt", ".pth", ".pt")):
            continue
        match = re.search(r"(?:^|[_-])epoch[=_-]?(\d+)(?=[._-]|$)", path.name)
        if match is None:
            continue
        epoch = int(match[1]) + 1
        if epoch % 10:
            continue
        if epoch in found:
            raise ValueError(f"Duplicate epoch {epoch}: {found[epoch]}, {path}. Select one training run.")
        found[epoch] = path
    return sorted(found.items())


def _completed(directory):
    try:
        manifest = _read_json(directory / "complete.json")
        if manifest.get("schema") != SCHEMA or not manifest.get("artifacts"):
            return None
        if not {"epoch", "checkpoint"}.issubset(manifest):
            return None
        for name, size in manifest["artifacts"].items():
            path = directory / name
            if not path.resolve().is_relative_to(directory.resolve()):
                return None
            if not path.is_file() or path.stat().st_size != size:
                return None
        return manifest
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _aggregate(output):
    """Rebuild only from committed epochs so crashes cannot duplicate rows."""
    rows = []
    observations = []
    directories = [path for path in output.glob("epoch-*")
                   if path.is_dir() and re.fullmatch(r"epoch-\d+", path.name)]
    for directory in sorted(directories, key=lambda path: int(path.name.split("-")[-1])):
        manifest = _completed(directory)
        if manifest is None:
            continue
        epoch = manifest["epoch"]
        for condition in CONDITIONS:
            for group in ("case", "organ"):
                path = directory / condition / "metrics" / f"{group}.csv"
                if not path.is_file():
                    continue
                with path.open(newline="", encoding="utf-8") as stream:
                    for row in csv.DictReader(stream):
                        rows.append({"epoch": epoch, "condition": condition, "group": group, **row})
        with (directory / "observations.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record["event"] in {"prediction", "runtime", "gamma"}:
                    observations.append({"epoch": epoch, **record})
    fields = ["epoch", "condition", "group", "case", "organ"]
    fields += sorted({key for row in rows for key in row} - set(fields))
    temporary = output / "epochs.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "epochs.csv")
    _write_json(output / "summary.json", observations)
    # Paired changes retain case/organ identity; positive HD change is worse,
    # whereas positive Dice change is better. No significance claim is made.
    baseline = {(row["epoch"], row["group"], row.get("case"), row.get("organ")): row
                for row in rows if row["condition"] == "normal"}
    deltas = []
    cohorts = {}
    metric_columns = [key for key in fields if key not in {"epoch", "condition", "group", "case", "organ"}]
    for row in rows:
        reference = baseline[(row["epoch"], row["group"], row.get("case"), row.get("organ"))]
        for metric in metric_columns:
            value, ref = float(row[metric]), float(reference[metric])
            deltas.append({"epoch": row["epoch"], "condition": row["condition"],
                           "group": row["group"], "case": row.get("case", ""),
                           "organ": row.get("organ", ""), "metric": metric,
                           "delta_vs_normal": value - ref if math.isfinite(value) and math.isfinite(ref) else ""})
            if row["group"] == "case":
                cohorts.setdefault((row["epoch"], row["condition"], metric), []).append(value)
    summaries = []
    for (epoch, condition, metric), values in cohorts.items():
        finite = [value for value in values if math.isfinite(value)]
        summaries.append({"epoch": epoch, "condition": condition, "metric": metric,
                          "finite_cases": len(finite), "nonfinite_cases": len(values) - len(finite),
                          "mean": statistics.fmean(finite) if finite else "",
                          "std_population": statistics.pstdev(finite) if finite else "",
                          "median": statistics.median(finite) if finite else "",
                          "min": min(finite) if finite else "", "max": max(finite) if finite else ""})
    for name, values, columns in (
        ("paired_deltas.csv", deltas, ["epoch", "condition", "group", "case", "organ", "metric", "delta_vs_normal"]),
        ("epoch_metrics.csv", summaries, ["epoch", "condition", "metric", "finite_cases", "nonfinite_cases",
                                          "mean", "std_population", "median", "min", "max"]),
    ):
        temporary = output / (name + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(values)
        temporary.replace(output / name)


class Observation:
    """One case at a time; retain bounded samples, never full feature volumes."""

    def __init__(self, directory, args, module):
        self.directory, self.args = directory, args
        self.stream = (directory / "observations.jsonl").open("x", encoding="utf-8")
        self.savers = {name: LogCallback(directory / name) for name in CONDITIONS}
        self.initialized = False
        self.emit({"event": "gamma", "effective": module.bottleneck.gamma.detach().cpu().tolist(),
                   "parameters": {name: value.detach().cpu().tolist()
                                  for name, value in module.bottleneck.named_parameters()
                                  if name == "gamma" or "parametrizations.gamma" in name},
                   "baseline": {"no_ppe": args.no_ppe, "no_mixing": args.no_mixing},
                   "feature_sample_limit": args.feature_samples,
                   "feature_comparison": "evenly spaced flattened elements of the same patch and stage",
                   "epoch_directory_unit": "completed epochs = Lightning checkpoint epoch + 1",
                   "timing_scope": "forward, feature observation, CPU transfer and stitching; excludes saving and metrics",
                   "probability_space": "pre-output-resampling, softmax of stitched logits",
                   "gamma_zero": "exact encoder residual bypass, not an unmodified pretrained decoder"})

    def emit(self, value):
        self.stream.write(json.dumps(_clean(value), ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()

    def _sample(self, value):
        flat = value.reshape(-1)
        count = min(flat.numel(), self.args.feature_samples)
        indices = torch.linspace(0, flat.numel() - 1, count, device=flat.device).long()
        return flat[indices].detach().to(device="cpu", dtype=torch.float32).clone()

    @staticmethod
    def _comparison(current, reference):
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

    def __call__(self, module, batch, batch_index):
        if not self.initialized:
            module.diagnostics_stream = self.stream
            try:
                module._start_diagnostic_session()
            finally:
                module.diagnostics_stream = None
            self.initialized = True
        image = batch[image_key]
        patches = module.split_to_patch(image)
        names = batch.get("name", [str(batch_index)])
        case = names[0] if isinstance(names, (tuple, list)) else str(names)
        references, first = {}, {}
        normal_logits = normal_prob = normal_labels = None
        flags = module.no_ppe, module.no_mixing
        callback = module.bottleneck.diagnostics_callback
        condition = "not_started"
        patch_inputs = {}
        encoder_norms = {}

        def feature(stage, value, position, metadata):
            if value is None:
                self.emit({"event": "bottleneck_call", "case": case,
                           "condition": condition, "metadata": metadata})
                return
            value = value.detach()
            if stage in {"encoder_features", "positional_encoding", "after_positional_encoding"}:
                value = value.transpose(1, 2)
            # Stats use CPU float32 to avoid overflowing half-precision reductions.
            current = value.to(device="cpu", dtype=torch.float32)
            sample = self._sample(current)
            key = stage, tuple(position)
            record = {"event": "feature", "case": case, "condition": condition,
                      "stage": stage, "position": list(position), "shape": list(current.shape),
                      **module._diagnostic_stats(current)}
            channels = current.reshape(current.shape[0], current.shape[1], -1)
            record["channel_mean"] = channels.mean((0, 2)).tolist()
            record["channel_rms"] = channels.double().square().mean((0, 2)).sqrt().tolist()
            if torch.isfinite(sample).all():
                record["sample_quantiles"] = dict(zip(
                    ("p01", "p05", "p50", "p95", "p99"),
                    torch.quantile(sample, torch.tensor([.01, .05, .5, .95, .99])).tolist()))
            if condition == "normal":
                references[key] = sample
                record["same_patch_vs_normal"] = {"status": "reference"}
            elif key in references:
                record["same_patch_vs_normal"] = self._comparison(sample, references[key])
            stage_key = condition, stage
            if stage_key in first:
                baseline, first_position = first[stage_key]
                record["vs_first_patch"] = {**self._comparison(sample, baseline),
                                            "reference_position": first_position}
            else:
                first[stage_key] = sample, list(position)
                record["vs_first_patch"] = {"status": "reference", "single_patch": len(patches) == 1}
            if stage == "encoder_features":
                patch_inputs[tuple(position)] = sample
                encoder_norms[tuple(position)] = current.double().norm().item()
            elif stage == "mamba_output" and tuple(position) in patch_inputs:
                scaled = sample * module.bottleneck.gamma.detach().cpu().float().reshape(-1)[0]
                residual = patch_inputs[tuple(position)]
                record["scaled_update_vs_encoder"] = self._comparison(scaled, residual)
                denominator = residual.double().norm().item()
                record["sample_scaled_update_l2_over_encoder_l2"] = scaled.double().norm().item() / denominator if denominator else None
                full_denominator = encoder_norms[tuple(position)]
                gamma = module.bottleneck.gamma.detach().abs().max().item()
                record["scaled_update_l2_over_encoder_l2"] = gamma * current.double().norm().item() / full_denominator if full_denominator else None
            elif stage == "after_residual" and tuple(position) in patch_inputs:
                record["output_vs_encoder"] = self._comparison(sample, patch_inputs.pop(tuple(position)))
                encoder_norms.pop(tuple(position), None)
            self.emit(record)

        try:
            module.bottleneck.diagnostics_callback = feature
            for condition in CONDITIONS:
                module.no_ppe = self.args.no_ppe or condition in {"no_ppe", "no_ppe_no_mixing"}
                module.no_mixing = self.args.no_mixing or condition in {"no_mixing", "no_ppe_no_mixing"}
                module.bottleneck.observation_bypass = condition == "gamma_zero"
                patch_inputs.clear()
                encoder_norms.clear()
                ordered = tuple(reversed(patches)) if condition == "reverse_order" else patches
                torch.cuda.synchronize(module.device)
                torch.cuda.reset_peak_memory_stats(module.device)
                started = time.perf_counter()
                results = module._collect_patch_outputs(
                    ordered, lambda index, label, out: feature("decoder_logits", out, ordered[index][2], None))
                if condition == "reverse_order":
                    # Keep CPU blending addition order fixed: this intervention
                    # should measure model sequence order, not sum roundoff.
                    by_position = {position: value for value, position in results}
                    results = tuple((by_position[position], position) for _, _, position in patches)
                    del by_position
                logits = stitch_logits(results, BlendMode.GAUSSIAN, output_size=image.shape[2:]).float()
                del results
                torch.cuda.synchronize(module.device)
                self.emit({"event": "runtime", "case": case, "condition": condition,
                           "seconds": time.perf_counter() - started,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(module.device),
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved(module.device),
                           "patch_count": len(patches), "patch_positions": [list(p[2]) for p in ordered],
                           "stitch_order": "original patch order for all conditions",
                           "no_ppe": module.no_ppe, "no_mixing": module.no_mixing})
                if not torch.isfinite(logits).all():
                    raise ValueError(f"Nonfinite logits: {case}, {condition}")
                probability = logits.softmax(1)
                labels = logits.argmax(1, keepdim=True)
                if condition == "normal":
                    normal_logits, normal_prob, normal_labels = logits, probability, labels
                record = {"event": "prediction", "case": case, "condition": condition,
                          "entropy_mean_nats": -(probability * probability.clamp_min(1e-30).log()).sum(1).mean().item(),
                          "confidence_mean": probability.max(1).values.mean().item(),
                          "top_two_margin_mean": probability.topk(2, dim=1).values.diff(dim=1).abs().mean().item(),
                          "logits_vs_normal": module._diagnostic_difference(logits, normal_logits),
                          "probabilities_vs_normal": module._diagnostic_difference(probability, normal_prob),
                          "changed_fraction": (labels != normal_labels).double().mean().item(),
                          "classes": []}
                # Chunk over spatial sites so probability divergences do not
                # create another full C x volume temporary allocation.
                pflat = probability.flatten(2)
                qflat = normal_prob.flatten(2)
                kl = js = 0.0
                for start in range(0, pflat.shape[-1], 32768):
                    p = pflat[:, :, start:start + 32768].double().clamp_min(1e-30)
                    q = qflat[:, :, start:start + 32768].double().clamp_min(1e-30)
                    middle = (p + q) * .5
                    kl += (p * (p.log() - q.log())).sum().item()
                    js += (.5 * (p * (p.log() - middle.log()) + q * (q.log() - middle.log()))).sum().item()
                record["kl_condition_to_normal_nats"] = kl / labels.numel()
                record["js_to_normal_nats"] = js / labels.numel()
                # Spacing is the effective inference grid, before val output inversion.
                record["grid_spacing_plan_units"] = list(module.plan.spacing)
                for organ in range(logits.shape[1]):
                    predicted, baseline = labels == organ, normal_labels == organ
                    count, base_count = predicted.sum().item(), baseline.sum().item()
                    intersection = (predicted & baseline).sum().item()
                    record["classes"].append({"class": organ, "voxels": count,
                        "fraction": count / labels.numel(), "probability_mean": probability[:, organ].mean().item(),
                        "vs_normal_dice": 2 * intersection / (count + base_count) if count + base_count else None,
                        "gained_voxels": (predicted & ~baseline).sum().item(),
                        "lost_voxels": (~predicted & baseline).sum().item()})
                self.emit(record)
                # Reuse exactly the val inversion and NIfTI writer for every condition.
                if condition != "normal":
                    self.savers[condition]._process_action("test", "out", batch, 0,
                                                          ("image", labels), True)
                del logits, probability, labels
                del pflat, qflat, p, q, middle
        finally:
            module.no_ppe, module.no_mixing = flags
            module.bottleneck.observation_bypass = False
            module.bottleneck.diagnostics_callback = callback
        return {"image": ("image", image.detach().cpu()),
                "out": ("image", normal_labels)}


def _worker(args, meta, custom_class):
    """Restore via the selected experiment's Custom class, including gamma parametrizations."""
    directory = args.output
    input_settings = _read_json(directory / "input.json")
    if _settings(args, meta) != input_settings:
        raise ValueError("Dataset or plan changed before worker startup")
    checkpoint_signature = _signature(args._checkpoint)
    checkpoint_hash = _sha256(args._checkpoint)
    checkpoint = torch.load(args._checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != args._epoch - 1:
        raise ValueError("Checkpoint filename epoch does not match saved Lightning epoch")
    del checkpoint

    class EpochValidation(custom_class):
        def parse_args(self, command):
            # Avoid the ordinary val timestamp suffix: this directory is an
            # exclusively owned attempt, and the parent commits it atomically.
            ArgumentAdaptor.parse_args(self, command)
            self.dataset_root = self.args.data
            self.cases = paired_cases(self.dataset_root / image_key, self.dataset_root / label_key)
            self.data = self.dataset_root / image_key
            self.save_path = directory / "normal"
            self.plan = Plan(self.args.plan_path)
            self.devices = [0]
            self.ckpt_path = self.args.ckpt

        def _build_trainer(self):
            return Trainer(accelerator="gpu", devices=[0], logger=False,
                           enable_checkpointing=False,
                           callbacks=[LogCallback(self.save_path)])

    command = ["val", str(args.data), str(directory), str(args.plan), "-c", str(args._checkpoint),
               "-w", str(args.workers)]
    if args.dice:
        command.append("--dice")
    if args.hd is not None:
        command.extend(["--hd", str(args.hd)])
    if args.no_ppe:
        command.append("--no-ppe")
    if args.no_mixing:
        command.append("--no-mixing")
    validation = EpochValidation(command, copy(meta))
    observation = Observation(directory, args, validation.module)
    sources = {Path(__file__).resolve(), Path(inspect.getfile(type(validation.module))).resolve(),
               Path(__file__).with_name("model.py").resolve(), Path(__file__).with_name("__init__.py").resolve(),
               Path(inspect.getfile(type(validation.module.bottleneck))).resolve(),
               Path(inspect.getfile(type(validation.module.bottleneck.bottleneck))).resolve()}
    observation.emit({"event": "source", "files_sha256": {str(path): _sha256(path) for path in sorted(sources)}})
    validation.module.observation_predictor = observation
    started = time.perf_counter()
    try:
        validation()
        for condition in CONDITIONS[1:]:
            write_metrics(validation.cases, directory / condition, epoch=0,
                          num_classes=validation._evaluation_num_classes(), dice=args.dice,
                          hd=args.hd, workers=args.workers)
        if _settings(args, meta) != input_settings:
            raise ValueError("Dataset or plan changed during observation")
        if _signature(args._checkpoint) != checkpoint_signature:
            raise ValueError("Checkpoint changed during observation; result will not be committed")
        manifest = {"schema": SCHEMA, "epoch": args._epoch, "lightning_epoch": args._epoch - 1,
                    "checkpoint": checkpoint_signature, "checkpoint_sha256": checkpoint_hash,
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "seconds_including_metrics": time.perf_counter() - started,
                    "conditions": list(CONDITIONS),
                    "artifacts": {str(path.relative_to(directory)): path.stat().st_size
                                  for path in directory.rglob("*")
                                  if path.is_file() and path.name not in {"worker.log", "complete.json"}}}
        observation.stream.close()
        _write_json(directory / "complete.json", manifest)
    finally:
        observation.stream.close()
        validation.module.observation_predictor = None


def observe(command, meta, custom_class):
    args = _parser(custom_class).parse_args(command)
    if args._checkpoint is not None:
        return _worker(args, meta, custom_class)
    if not args.trained.is_dir():
        raise FileNotFoundError(args.trained)
    if args.data.is_relative_to(args.output) or args.trained.is_relative_to(args.output):
        raise ValueError("Output must not contain the input dataset or training directory")
    settings = _settings(args, meta)
    args.output.mkdir(parents=True, exist_ok=True)
    # OS advisory locks are released on crash; a persistent lock file is safe.
    lock = (args.output / ".observe.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        lock.close()
        raise RuntimeError(f"Another observer owns {args.output}") from exc
    child = None
    try:
        config = args.output / "observe.json"
        if config.exists():
            if _read_json(config) != settings:
                raise ValueError("Output settings/dataset differ; choose another output directory")
        else:
            if any(args.output.glob("epoch-*")):
                raise ValueError("Existing epochs have no observe.json; choose another output directory")
            _write_json(config, settings)
        _aggregate(args.output)
        seen = {}
        while True:
            try:
                if _settings(args, meta) != settings:
                    raise ValueError("Dataset or plan changed; choose a new output directory")
                checkpoints = _checkpoints(args.trained)
                for epoch, checkpoint in checkpoints:
                    signature = _signature(checkpoint)
                    destination = args.output / f"epoch-{epoch}"
                    if destination.exists():
                        completed = _completed(destination)
                        if completed is None:
                            # Keep interrupted artifacts for inspection; retry
                            # into a new exclusively owned attempt directory.
                            archive = args.output / ".attempts" / f"incomplete-{destination.name}-{uuid.uuid4().hex}"
                            archive.parent.mkdir(parents=True, exist_ok=True)
                            destination.rename(archive)
                            print(f"Preserved incomplete epoch in {archive}", flush=True)
                        else:
                            if completed["checkpoint"] != signature:
                                raise ValueError(f"Completed checkpoint was replaced: {checkpoint}")
                            continue
                    # Must be unchanged across two scans, separated by interval.
                    previous = seen.get(str(checkpoint))
                    if previous is None or previous[0] != signature:
                        seen[str(checkpoint)] = (signature, time.monotonic())
                        continue
                    if time.monotonic() - previous[1] < args.interval:
                        continue
                    gpu = _idle_gpu(args.devices)
                    if gpu is None:
                        print("No idle GPU; waiting.", flush=True)
                        break
                    # Repeat telemetry immediately before launching the worker.
                    rechecked = _idle_gpu([gpu["logical_index"]])
                    if rechecked is None or rechecked["uuid"] != gpu["uuid"]:
                        break
                    attempt = args.output / ".attempts" / f"epoch-{epoch}-{uuid.uuid4().hex}"
                    attempt.mkdir(parents=True, exist_ok=False)
                    _write_json(attempt / "gpu.json", gpu)
                    _write_json(attempt / "input.json", settings)
                    worker_command = [sys.executable, str(Path(__file__).resolve().parents[2] / "main.py"),
                                      "munet", "custom"]
                    if meta.experiment_name is not None:
                        worker_command.extend(["--experiment_name", meta.experiment_name])
                    worker_command.extend(["observe", str(args.data), str(args.trained), str(args.plan),
                                           str(attempt), "--_checkpoint", str(checkpoint), "--_epoch", str(epoch),
                                           "-w", str(args.workers), "--feature-samples", str(args.feature_samples)])
                    if args.dice:
                        worker_command.append("--dice")
                    if args.hd is not None:
                        worker_command.extend(["--hd", str(args.hd)])
                    for flag in ("no_ppe", "no_mixing"):
                        if getattr(args, flag):
                            worker_command.append("--" + flag.replace("_", "-"))
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID")
                    print(f"Observing epoch {epoch} on {gpu['uuid']}: {attempt}", flush=True)
                    with (attempt / "worker.log").open("x", encoding="utf-8") as log:
                        child = subprocess.Popen(worker_command, env=environment, stdout=log,
                                                 stderr=subprocess.STDOUT,
                                                 start_new_session=os.name != "nt")
                        next_status = time.monotonic() + 60
                        while child.poll() is None:
                            time.sleep(0.5)
                            if time.monotonic() >= next_status:
                                print(f"Epoch {epoch} running on {gpu['uuid']}; log: {attempt / 'worker.log'}", flush=True)
                                next_status = time.monotonic() + 60
                        code = child.returncode
                        child = None
                    if code != 0 or not (attempt / "complete.json").is_file():
                        print(f"Epoch {epoch} failed (exit={code}); see {attempt / 'worker.log'}. Retry next round.", flush=True)
                        continue
                    if _signature(checkpoint) != signature:
                        print(f"Checkpoint changed; retaining uncommitted attempt {attempt}", flush=True)
                        seen.pop(str(checkpoint), None)
                        break
                    attempt.rename(destination)
                    _aggregate(args.output)
                    print(f"Completed epoch {epoch}: {destination}", flush=True)
                    gc.collect()
            except (subprocess.SubprocessError, FileNotFoundError) as exc:
                print(f"Observation deferred: {exc}", flush=True)
            print(f"Next scan in {args.interval:g}s; Ctrl+C to stop.", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopping observer; completed epochs are retained.", flush=True)
    finally:
        if child is not None and child.poll() is None:
            if os.name == "nt":
                child.terminate()
            else:
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    child.kill()
                else:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        lock.close()
