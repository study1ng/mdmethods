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
import tempfile
import time
import uuid
from copy import copy
from datetime import datetime, timezone
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import platform
from importlib.metadata import PackageNotFoundError, version

import torch
from lightning import Trainer
from monai.data import MetaTensor

from experiments import ArgumentAdaptor
from experiments.config import image_key, label_key
from experiments.munet.evaluation import paired_cases
from experiments.munet.observe_cpu import (
    BOTTLENECK_STAGES, METRIC_GROUPS, comparison, full_feature_stats,
    initialize_worker, process_case, write_metric_rows,
)
from experiments.plan import Plan
from experiments.utils import resolved_path


SCHEMA = 2
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


def _nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def _training_roots(value):
    parts = value.split(",")
    if any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError("trained must contain nonempty comma-separated directories")
    return tuple(sorted({resolved_path(part.strip()) for part in parts}))


def _sample_indices(elements, limit, *, device="cpu"):
    """Inclusive endpoints, integer arithmetic only, O(limit) storage.

    Python integers also avoid overflow in i * (elements - 1). A float32
    linspace can round the last valid index up past the end for large tensors.
    """
    if elements < 0 or limit < 1:
        raise ValueError("elements must be nonnegative and limit must be positive")
    count = min(elements, limit)
    values = ([index * (elements - 1) // (count - 1) for index in range(count)]
              if count > 1 else [0] if count == 1 else [])
    return torch.tensor(values, dtype=torch.long, device=device)


def _parser(custom_class):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=resolved_path)
    parser.add_argument("trained", type=_training_roots,
                        help="One training directory or comma-separated continuation directories.")
    parser.add_argument("plan", type=resolved_path)
    parser.add_argument("output", type=resolved_path)
    parser.add_argument("-mh", "--module-help", action="help")
    parser.add_argument("-i", "--interval", type=_positive, default=60)
    parser.add_argument("-w", "--workers", type=_positive_int, default=4)
    parser.add_argument("--loader-workers", type=_nonnegative_int, default=4,
                        help="Inference DataLoader workers; 0 disables loader multiprocessing.")
    parser.add_argument("--temp-dir", type=resolved_path,
                        default=Path("/tmp") if os.name != "nt" else resolved_path(tempfile.gettempdir()),
                        help="Existing local temporary directory for worker IPC; Linux default: /tmp.")
    parser.add_argument("--dice", action="store_true")
    parser.add_argument("--hd", type=custom_class._hd_percentile, default=None)
    parser.add_argument("-d", "--devices", type=int, nargs="+", default=None,
                        help="Candidate logical CUDA indices; default: all visible GPUs.")
    parser.add_argument("--no-ppe", action="store_true", help="Disable PPE in the baseline.")
    parser.add_argument("--no-mixing", action="store_true", help="Disable mixing in the baseline.")
    parser.add_argument("--feature-samples", type=_positive_int, default=4096,
                        help="Maximum evenly spaced feature elements retained per patch/stage.")
    parser.add_argument("--observe-metrics", choices=(*METRIC_GROUPS, "all"), nargs="+",
                        default=["basic"], help="Enabled groups; quality alone enables Dice, HD remains opt-in.")
    parser.add_argument("--save-predictions", choices=("none", "normal", "all"), default="none")
    parser.add_argument("--save-orthogonal", choices=("none", "normal", "all"), default="none")
    parser.add_argument("--max-pending", type=_positive_int, default=2,
                        help="Maximum submitted CPU jobs, running plus queued (default: 2).")
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


def _cpu_metadata(value):
    """Transform histories may contain tensors too; never send CUDA metadata."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_metadata(item) for item in value)
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
    return {"schema": SCHEMA, "data": str(args.data), "trained": [str(root) for root in args.trained],
            "plan_sha256": _sha256(args.plan), "experiment": meta.experiment_name,
            "dice": args.dice, "hd": args.hd, "feature_samples": args.feature_samples,
            "observe_metrics": args.observe_metrics,
            "save_predictions": args.save_predictions, "save_orthogonal": args.save_orthogonal,
            "no_ppe": args.no_ppe, "no_mixing": args.no_mixing,
            "conditions": list(CONDITIONS),
            "cases": {name: {key: _signature(path) for key, path in case.items()}
                      for name, case in cases.items()}}


def _compatible_settings(previous, current):
    """Accept legacy single-root settings and additions to the search roots."""
    previous, current = dict(previous), dict(current)
    old_roots = previous.pop("trained")
    new_roots = current.pop("trained")
    if isinstance(old_roots, str):
        old_roots = [old_roots]
    return previous == current and set(old_roots).issubset(new_roots)


def _worker_environment(args, gpu):
    # These must be set before launching Python, before tempfile or
    # multiprocessing can cache an inherited shared-filesystem TMPDIR.
    return dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
                TMPDIR=str(args.temp_dir), TEMP=str(args.temp_dir), TMP=str(args.temp_dir),
                PYTHONUNBUFFERED="1")


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


def _checkpoints(roots):
    """Only immutable numbered periodic snapshots, never the mutable last.ckpt."""
    found = {}
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or not path.name.endswith((".ckpt", ".pth", ".pt")):
                continue
            match = re.search(r"(?:^|[_-])epoch[=_-]?(\d+)(?=[._-]|$)", path.name)
            if match is None:
                continue
            epoch = int(match[1]) + 1
            if epoch % 10:
                continue
            found.setdefault(epoch, set()).add(path.resolve())
    return [(epoch, tuple(sorted(paths))) for epoch, paths in sorted(found.items())]


def _stable_candidates(paths, seen, interval):
    ready = True
    now = time.monotonic()
    for path in paths:
        signature = _signature(path)
        previous = seen.get(str(path))
        if previous is None or previous[0] != signature:
            seen[str(path)] = signature, now
            ready = False
        elif now - previous[1] < interval:
            ready = False
    return ready


def _cached_digest(path, digests):
    signature = _signature(path)
    key = str(path), signature["size"], signature["mtime_ns"]
    if key not in digests:
        digest = _sha256(path)
        if _signature(path) != signature:
            raise OSError(f"Checkpoint changed while hashing: {path}")
        digests[key] = digest
    return digests[key]


def _select_checkpoint(epoch, paths, digests, completed=None):
    """Called only after every candidate is stable; identical copies count once."""
    if len(paths) > 1:
        hashes = {_cached_digest(path, digests) for path in paths}
        if len(hashes) != 1:
            raise ValueError(f"Conflicting checkpoints for epoch {epoch}: {paths}")
    if completed is not None:
        for path in paths:
            if _signature(path) == completed["checkpoint"]:
                return path
        path = paths[0]
        if _cached_digest(path, digests) != completed.get("checkpoint_sha256"):
            raise ValueError(f"Completed checkpoint was replaced at epoch {epoch}: {path}")
        return path
    return paths[0]


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
                if record["event"] in {"prediction", "runtime", "gamma", "cpu_runtime",
                                       "cpu_case_runtime", "pipeline"}:
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
    """One GPU producer and bounded CPU jobs; no model crosses the process boundary."""

    def __init__(self, directory, args, module, cases=None, num_classes=None):
        self.directory, self.args = directory, args
        self.cases = cases if cases is not None else paired_cases(args.data / image_key, args.data / label_key)
        self.num_classes = num_classes
        self.stream = (directory / "observations.jsonl").open("x", encoding="utf-8")
        self.pending = deque()
        self.metrics = {}
        self.full_encoder_norms = {}
        self.queue_wait_seconds = 0.0
        self.gamma = module.bottleneck.gamma.detach().cpu().float().reshape(-1)[0].item()
        self.pool = ProcessPoolExecutor(max_workers=args.workers,
                                        mp_context=get_context("spawn"),
                                        initializer=initialize_worker)
        self.emit({"event": "gamma", "effective": module.bottleneck.gamma.detach().cpu().tolist(),
                   "parameters": {name: value.detach().cpu().tolist()
                                  for name, value in module.bottleneck.named_parameters()
                                  if name == "gamma" or "parametrizations.gamma" in name},
                   "baseline": {"no_ppe": args.no_ppe, "no_mixing": args.no_mixing},
                   "feature_sample_limit": args.feature_samples,
                   "feature_comparison": "evenly spaced flattened elements of the same patch and stage",
                   "epoch_directory_unit": "completed epochs = Lightning checkpoint epoch + 1",
                   "timing_scope": "GPU runtime: forward, observation copies and queue backpressure; CPU timings separate",
                   "probability_space": "pre-output-resampling, softmax of stitched logits",
                   "gamma_zero": "exact encoder residual bypass, not an unmodified pretrained decoder"})
        self.emit({"event": "diagnostic_environment", "schema_version": SCHEMA,
                   "python": platform.python_version(), "torch": torch.__version__,
                   "cuda_version": torch.version.cuda,
                   "matmul_precision": torch.get_float32_matmul_precision(),
                   "cudnn_benchmark": torch.backends.cudnn.benchmark,
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "observe_metrics": args.observe_metrics,
                   "save_predictions": args.save_predictions, "save_orthogonal": args.save_orthogonal})
        packages = {}
        for package in ("torch", "lightning", "monai", "mamba-ssm", "causal-conv1d"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = "not_found"
        self.emit({"event": "packages", "versions": packages})
        if "full-stats" in args.observe_metrics:
            for name, parameter in module.bottleneck.named_parameters():
                self._submit(full_feature_stats,
                             {"event": "parameter_stats", "parameter": name,
                              "shape": list(parameter.shape), "dtype": str(parameter.dtype),
                              "requires_grad": parameter.requires_grad},
                             parameter.detach().to(device="cpu", dtype=torch.float32, copy=True))

    def emit(self, value):
        self.stream.write(json.dumps(_clean(value), ensure_ascii=False, allow_nan=False) + "\n")

    def _sample(self, value):
        if isinstance(value, MetaTensor):
            value = value.as_tensor()
        indices = _sample_indices(value.numel(), self.args.feature_samples, device=value.device)
        if value.is_contiguous():
            sample = value.view(-1)[indices]
        else:
            # Flattening a non-contiguous feature may allocate a whole GPU volume.
            # Convert only the sampled flat indices to logical tensor coordinates.
            coordinates = []
            for size in reversed(value.shape):
                coordinates.append(indices.remainder(size))
                indices = indices.div(size, rounding_mode="floor")
            sample = value[tuple(reversed(coordinates))]
        return sample.detach().to(device="cpu", dtype=torch.float32).clone()

    _comparison = staticmethod(comparison)

    def _accept(self, result):
        for record in result["records"]:
            if record["event"] == "feature_full_stats":
                key = record["case"], record["condition"], tuple(record["position"])
                stage = record["stage"]
                if stage == "encoder_features":
                    self.full_encoder_norms[key] = record["feature_l2"]
                elif stage == "mamba_output" and key in self.full_encoder_norms:
                    denominator = self.full_encoder_norms[key]
                    record["scaled_update_l2_over_encoder_l2"] = (
                        abs(self.gamma) * record["feature_l2"] / denominator if denominator else None)
                elif stage == "after_residual":
                    self.full_encoder_norms.pop(key, None)
            self.emit(record)
        for condition, rows in result.get("metrics", {}).items():
            self.metrics.setdefault(condition, {})[rows[0]["case"]] = rows
        self.stream.flush()

    def _take(self):
        started = time.perf_counter()
        result = self.pending.popleft().result()
        self.queue_wait_seconds += time.perf_counter() - started
        self._accept(result)

    def _make_room(self):
        while self.pending and self.pending[0].done():
            self._take()
        while len(self.pending) >= self.args.max_pending:
            self._take()

    def _submit(self, function, *arguments):
        self._make_room()
        self.pending.append(self.pool.submit(function, *arguments))

    def finish(self):
        while self.pending:
            self._take()
        if self.args.dice or self.args.hd is not None:
            for condition in CONDITIONS:
                if set(self.metrics.get(condition, {})) != set(self.cases):
                    raise ValueError(f"Incomplete CPU metrics: {condition}")
            write_metric_rows(self.directory, self.cases, self.metrics, self.args.dice, self.args.hd)
        self.emit({"event": "pipeline", "queue_wait_seconds": self.queue_wait_seconds,
                   "cpu_workers": self.args.workers, "max_pending": self.args.max_pending})
        self.stream.flush()

    def close(self):
        for future in self.pending:
            future.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.stream.close()

    def __call__(self, module, batch, batch_index):
        # Bound outstanding cases before allocating another case's six outputs.
        self._make_room()
        image = batch[image_key]
        if image.shape[0] != 1:
            raise ValueError("Observation requires one case per inference batch")
        patches = module.split_to_patch(image)
        if not patches:
            raise ValueError("Observation cannot predict an image with no patches")
        names = batch.get("name", [str(batch_index)])
        case = names[0] if isinstance(names, (tuple, list)) else str(names)
        groups = set(self.args.observe_metrics)
        features, samples, outputs = [], [], []
        sample_offset = 0
        flags = module.no_ppe, module.no_mixing
        callback = module.bottleneck.diagnostics_callback
        previous_bypass = getattr(module.bottleneck, "observation_bypass", False)
        condition = "not_started"
        observe_features = bool(groups & {"bottleneck", "internal", "full-stats"})

        def feature(stage, value, position, metadata):
            nonlocal sample_offset
            if value is None:
                self.emit({"event": "bottleneck_call", "case": case,
                           "condition": condition, "metadata": metadata})
                return
            if not (groups & {"internal", "full-stats"}) and stage not in BOTTLENECK_STAGES:
                return
            value = value.detach()
            if stage in {"encoder_features", "positional_encoding", "after_positional_encoding"}:
                value = value.transpose(1, 2)
            sample = self._sample(value)
            record = {"event": "feature", "case": case, "condition": condition,
                      "stage": stage, "position": list(position), "shape": list(value.shape),
                      "sample_slice": (sample_offset, sample.numel())}
            features.append(record)
            samples.append(sample)
            sample_offset += sample.numel()
            if "full-stats" in groups:
                # Snapshot before the forward's in-place residual operations.
                # The bounded pool releases full features after returning compact statistics.
                self._make_room()
                current = value.to(device="cpu", dtype=torch.float32, copy=True)
                if isinstance(current, MetaTensor):
                    current = current.as_tensor()
                full_record = {key: item for key, item in record.items() if key != "sample_slice"}
                full_record["event"] = "feature_full_stats"
                self._submit(full_feature_stats, full_record, current)

        try:
            module.bottleneck.diagnostics_callback = feature if observe_features else None
            for condition in CONDITIONS:
                module.no_ppe = self.args.no_ppe or condition in {"no_ppe", "no_ppe_no_mixing"}
                module.no_mixing = self.args.no_mixing or condition in {"no_mixing", "no_ppe_no_mixing"}
                module.bottleneck.observation_bypass = condition == "gamma_zero"
                ordered = tuple(reversed(patches)) if condition == "reverse_order" else patches
                torch.cuda.synchronize(module.device)
                torch.cuda.reset_peak_memory_stats(module.device)
                started, wait_before = time.perf_counter(), self.queue_wait_seconds
                decoder_callback = (lambda index, label, out:
                                    feature("decoder_logits", out, ordered[index][2], None))
                results = module._collect_patch_outputs(
                    ordered, decoder_callback if groups & {"internal", "full-stats"} else None)
                if condition == "reverse_order":
                    by_position = {position: value for value, position in results}
                    results = tuple((by_position[position], position) for _, _, position in patches)
                    del by_position
                torch.cuda.synchronize(module.device)
                self.emit({"event": "runtime", "case": case, "condition": condition,
                           "seconds": time.perf_counter() - started,
                           "queue_wait_seconds": self.queue_wait_seconds - wait_before,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(module.device),
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved(module.device),
                           "patch_count": len(patches), "patch_positions": [list(p[2]) for p in ordered],
                           "stitch_order": "original patch order for all conditions",
                           "no_ppe": module.no_ppe, "no_mixing": module.no_mixing})
                # Shared CPU storage avoids serializing large tensor payloads into
                # each process pipe. Never share the original GPU tensors.
                shared_results = []
                for value, position in results:
                    if isinstance(value, MetaTensor):
                        value = value.as_tensor()
                    shared_results.append((value.share_memory_(), position))
                outputs.append((condition, shared_results))
                del results
                self.stream.flush()
        finally:
            module.no_ppe, module.no_mixing = flags
            module.bottleneck.observation_bypass = previous_bypass
            module.bottleneck.diagnostics_callback = callback
        needs_reference = (self.args.save_predictions != "none" or self.args.save_orthogonal != "none"
                           or self.args.dice or self.args.hd is not None)
        cpu_batch = None
        if needs_reference:
            # Preserve MetaTensor transform history for the existing label inversion.
            reference = image.detach().to(device="cpu", copy=True)
            if isinstance(reference, MetaTensor):
                reference.meta = _cpu_metadata(reference.meta)
                reference.applied_operations = _cpu_metadata(reference.applied_operations)
            cpu_batch = {image_key: reference, "name": names}
        packed_samples = torch.cat(samples) if samples else torch.empty(0)
        self._submit(process_case, {
            "case": case, "groups": self.args.observe_metrics, "outputs": outputs,
            "features": features, "samples": packed_samples.share_memory_(),
            "gamma": self.gamma, "patch_count": len(patches),
            "output_size": tuple(image.shape[2:]), "spacing": list(module.plan.spacing),
            "batch": cpu_batch, "source": self.cases[case], "directory": self.directory,
            "num_classes": self.num_classes, "dice": self.args.dice, "hd": self.args.hd,
            "save_predictions": self.args.save_predictions, "save_orthogonal": self.args.save_orthogonal,
        })
        # Saving and metrics are exclusively owned by CPU jobs, not LogCallback.
        return {}


def _worker(args, meta, custom_class):
    """Restore via the selected experiment's Custom class, including gamma parametrizations."""
    directory = args.output
    tempfile.tempdir = str(args.temp_dir)
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
                           callbacks=[])

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
    validation.datamodule.num_workers = args.loader_workers
    observation = Observation(directory, args, validation.module, validation.cases,
                              validation._evaluation_num_classes())
    observation.emit({"event": "execution", "temp_dir": str(args.temp_dir),
                      "loader_workers": args.loader_workers, "cpu_workers": args.workers,
                      "max_pending": args.max_pending, "cpu_job_unit": "case (six conditions) or full feature statistics",
                      "observe_metrics": args.observe_metrics,
                      "save_predictions": args.save_predictions, "save_orthogonal": args.save_orthogonal,
                      "sampling_rule": "integer floor(i * (elements - 1) / (count - 1))"})
    sources = {Path(__file__).resolve(), Path(inspect.getfile(type(validation.module))).resolve(),
               Path(__file__).with_name("observe_cpu.py").resolve(),
               Path(__file__).with_name("evaluation.py").resolve(),
               Path(__file__).with_name("model.py").resolve(), Path(__file__).with_name("__init__.py").resolve(),
               Path(inspect.getfile(type(validation.module.bottleneck))).resolve(),
               Path(inspect.getfile(type(validation.module.bottleneck.bottleneck))).resolve()}
    observation.emit({"event": "source", "files_sha256": {str(path): _sha256(path) for path in sorted(sources)}})
    validation.module.observation_predictor = observation
    started = time.perf_counter()
    try:
        # The custom class still restores the selected experiment. Only observe's
        # test output handling changes: no synchronous val saving/evaluation.
        validation.trainer.test(model=validation.module, datamodule=validation.datamodule)
        observation.finish()
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
        _write_json(directory / "complete.json", manifest)
    finally:
        observation.close()
        validation.module.observation_predictor = None


def observe(command, meta, custom_class):
    args = _parser(custom_class).parse_args(command)
    groups = set(args.observe_metrics)
    if "all" in groups:
        groups = set(METRIC_GROUPS)
    if "quality" in groups and not args.dice and args.hd is None:
        args.dice = True
    if args.dice or args.hd is not None:
        groups.add("quality")
    args.observe_metrics = [name for name in METRIC_GROUPS if name in groups]
    if not args.temp_dir.is_dir() or not os.access(args.temp_dir, os.W_OK | os.X_OK):
        raise ValueError(f"--temp-dir must be an existing writable/searchable local directory: {args.temp_dir}")
    if args._checkpoint is not None:
        return _worker(args, meta, custom_class)
    for root in args.trained:
        if not root.is_dir():
            raise FileNotFoundError(root)
    if args.data.is_relative_to(args.output) or any(root.is_relative_to(args.output) for root in args.trained):
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
            previous = _read_json(config)
            if not _compatible_settings(previous, settings):
                raise ValueError("Output settings/dataset differ or search roots were removed; choose another output directory")
            if previous != settings:
                _write_json(config, settings)
        else:
            if any(args.output.glob("epoch-*")):
                raise ValueError("Existing epochs have no observe.json; choose another output directory")
            _write_json(config, settings)
        _aggregate(args.output)
        seen = {}
        digests = {}
        while True:
            try:
                if _settings(args, meta) != settings:
                    raise ValueError("Dataset or plan changed; choose a new output directory")
                checkpoints = _checkpoints(args.trained)
                for epoch, paths in checkpoints:
                    # Also wait for duplicate copies before comparing hashes;
                    # a partially copied checkpoint must not cause a conflict.
                    if not _stable_candidates(paths, seen, args.interval):
                        continue
                    destination = args.output / f"epoch-{epoch}"
                    completed = _completed(destination) if destination.exists() else None
                    checkpoint = _select_checkpoint(epoch, paths, digests, completed)
                    signature = _signature(checkpoint)
                    if destination.exists():
                        if completed is None:
                            # Keep interrupted artifacts for inspection; retry
                            # into a new exclusively owned attempt directory.
                            archive = args.output / ".attempts" / f"incomplete-{destination.name}-{uuid.uuid4().hex}"
                            archive.parent.mkdir(parents=True, exist_ok=True)
                            destination.rename(archive)
                            print(f"Preserved incomplete epoch in {archive}", flush=True)
                        else:
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
                    worker_command.extend(["observe", str(args.data), ",".join(str(root) for root in args.trained), str(args.plan),
                                           str(attempt), "--_checkpoint", str(checkpoint), "--_epoch", str(epoch),
                                           "-w", str(args.workers), "--feature-samples", str(args.feature_samples),
                                           "--loader-workers", str(args.loader_workers), "--temp-dir", str(args.temp_dir),
                                           "--max-pending", str(args.max_pending),
                                           "--save-predictions", args.save_predictions,
                                           "--save-orthogonal", args.save_orthogonal,
                                           "--observe-metrics", *args.observe_metrics])
                    if args.dice:
                        worker_command.append("--dice")
                    if args.hd is not None:
                        worker_command.extend(["--hd", str(args.hd)])
                    for flag in ("no_ppe", "no_mixing"):
                        if getattr(args, flag):
                            worker_command.append("--" + flag.replace("_", "-"))
                    environment = _worker_environment(args, gpu)
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
            except (subprocess.SubprocessError, OSError) as exc:
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
