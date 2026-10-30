"""Observe pretrained segmentation checkpoints using the original MONAI inference."""

import argparse
import csv
import gc
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from multiprocessing import get_context

import torch
from monai.data import MetaTensor
from monai.inferers import sliding_window_inference

from experiments.config import image_key, label_key
from experiments.munet.evaluation import paired_cases
from experiments.munet.observe import (
    Observation as BaseObservation, _IpcDirectory, _clean, _cleanup_epoch_ipc,
    _completed, _compatible_settings, _cpu_metadata, _ignore_sigint, _idle_gpu,
    _positive, _positive_int, _nonnegative_int, _read_json, _select_checkpoint,
    _sha256, _signature, _stable_candidates, _stop_epoch, _training_roots,
    _worker_environment, _write_json, _checkpoints,
)
from experiments.munet.observe_cpu import full_feature_stats, initialize_worker, write_metric_rows
from experiments.nets.ubimamba import BiMambaBlock
from experiments.plan import Plan
from experiments.pretrained_seg import DataModule, Model
from experiments.pretrained_seg.observe_cpu import process_case
from experiments.utils import resolved_path

SCHEMA = 2
CONDITIONS = ("normal",)
METRIC_GROUPS = ("basic", "quality", "confidence", "distribution", "bottleneck", "internal", "full-stats")


class Observation(BaseObservation):
    """Reuse the bounded disk-backed CPU queue with a MONAI GPU producer."""

    def __init__(self, directory, args, module, cases, num_classes):
        self.directory, self.args, self.cases = directory, args, cases
        self.num_classes = num_classes
        self.stream = (directory / "observations.jsonl").open("x", encoding="utf-8")
        self.pending, self.metrics = deque(), {}
        self.full_encoder_norms = {}
        self.gamma = 1.0
        self.queue_wait_seconds = self.disk_write_seconds = self.disk_read_seconds = 0.0
        self.ipc_directory = _IpcDirectory(args._ipc_dir or args.temp_dir)
        self.pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"),
                                        initializer=initialize_worker)
        self.emit({"event": "diagnostic_environment", "schema_version": SCHEMA,
                   "python": platform.python_version(), "torch": torch.__version__,
                   "cuda_version": torch.version.cuda,
                   "matmul_precision": torch.get_float32_matmul_precision(),
                   "cudnn_benchmark": torch.backends.cudnn.benchmark,
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "observe_metrics": args.observe_metrics,
                   "feature_sample_limit": args.feature_samples,
                   "conditions": list(CONDITIONS),
                   "inference": "MONAI sliding_window_inference, overlap=0.5, gaussian, replicate",
                   "probability_space": "softmax of MONAI stitched logits before output resampling"})
        packages = {}
        for package in ("torch", "lightning", "monai", "mamba-ssm", "causal-conv1d"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = "not_found"
        self.emit({"event": "packages", "versions": packages})

    def finish(self):
        while self.pending:
            self._take()
        if self.args.dice or self.args.hd is not None:
            if set(self.metrics.get("normal", {})) != set(self.cases):
                raise ValueError("Incomplete CPU metrics: normal")
            write_metric_rows(self.directory, self.cases, self.metrics, self.args.dice, self.args.hd)
        self.emit({"event": "pipeline", "queue_wait_seconds": self.queue_wait_seconds,
                   "disk_write_seconds": self.disk_write_seconds,
                   "disk_read_seconds": self.disk_read_seconds,
                   "cpu_workers": self.args.workers, "max_pending": self.args.max_pending})
        self.stream.flush()

    def __call__(self, module, batch, batch_index):
        self._make_room()
        image = batch[image_key]
        if image.shape[0] != 1:
            raise ValueError("Observation requires one case per batch")
        names = batch["name"]
        case = names[0] if isinstance(names, (tuple, list)) else str(names)
        groups = set(self.args.observe_metrics)
        features, samples, handles, callbacks = [], [], [], []
        offset, patch_index = 0, -1

        def feature(stage, value):
            nonlocal offset
            value = value.detach()
            sample = self._sample(value)
            record = {"event": "feature", "case": case, "condition": "normal",
                      "stage": stage, "position": [patch_index], "shape": list(value.shape),
                      "sample_slice": (offset, sample.numel())}
            features.append(record)
            samples.append(sample)
            offset += sample.numel()
            if "full-stats" in groups:
                self._make_room()
                current = value.to(device="cpu", dtype=torch.float32, copy=True)
                if isinstance(current, MetaTensor):
                    current = current.as_tensor()
                full_record = {key: item for key, item in record.items() if key != "sample_slice"}
                full_record["event"] = "feature_full_stats"
                self._submit(full_feature_stats, full_record, current)

        def pre_hook(name):
            return lambda block, inputs: feature(f"{name}/input", inputs[0])

        def post_hook(name):
            return lambda block, inputs, output: feature(f"{name}/output", output)

        def predictor(patch):
            nonlocal patch_index
            patch_index += 1
            output = module(patch)
            if groups & {"internal", "full-stats"}:
                feature("decoder_logits", output)
            return output

        torch.cuda.synchronize(module.device)
        torch.cuda.reset_peak_memory_stats(module.device)
        started, wait_before = time.perf_counter(), self.queue_wait_seconds
        try:
            if groups & {"bottleneck", "internal", "full-stats"}:
                blocks = [(name, block) for name, block in module.unet.named_modules()
                          if isinstance(block, BiMambaBlock)]
                if not blocks:
                    raise ValueError("Feature observation requires a checkpoint containing BiMamba blocks")
                for name, block in blocks:
                    handles.append(block.register_forward_pre_hook(pre_hook(name)))
                    handles.append(block.register_forward_hook(post_hook(name)))
                    callbacks.append((block, block.diagnostics_callback))
                    if groups & {"internal", "full-stats"}:
                        block.diagnostics_callback = lambda stage, value, name=name: feature(
                            f"{name}/{stage}", value.transpose(1, 2))
                    else:
                        block.diagnostics_callback = None
                if "full-stats" in groups and batch_index == 0:
                    for name, block in blocks:
                        for parameter_name, parameter in block.named_parameters():
                            self._submit(full_feature_stats,
                                         {"event": "parameter_stats", "case": case,
                                          "parameter": f"{name}.{parameter_name}",
                                          "shape": list(parameter.shape), "dtype": str(parameter.dtype)},
                                         parameter.detach().to(device="cpu", dtype=torch.float32, copy=True))
            logits = sliding_window_inference(
                image, roi_size=module.plan.patch_size, sw_batch_size=1, predictor=predictor,
                overlap=0.5, mode="gaussian", progress=None, device=module.device,
                padding_mode="replicate")
            torch.cuda.synchronize(module.device)
            self.emit({"event": "runtime", "case": case, "condition": "normal",
                       "seconds": time.perf_counter() - started,
                       "queue_wait_seconds": self.queue_wait_seconds - wait_before,
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(module.device),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(module.device),
                       "patch_count": patch_index + 1})
        finally:
            for handle in handles:
                handle.remove()
            for block, callback in callbacks:
                block.diagnostics_callback = callback
        logits = logits.detach().to(device="cpu", dtype=torch.float32)
        if isinstance(logits, MetaTensor):
            logits = logits.as_tensor()
        reference = None
        if (self.args.dice or self.args.hd is not None or self.args.save_predictions != "none"
                or self.args.save_orthogonal != "none"):
            reference = image.detach().to(device="cpu", copy=True)
            if isinstance(reference, MetaTensor):
                reference.meta = _cpu_metadata(reference.meta)
                reference.applied_operations = _cpu_metadata(reference.applied_operations)
            reference = {image_key: reference, "name": names}
        self._submit(process_case, {
            "case": case, "groups": self.args.observe_metrics,
            "outputs": [("normal", [logits])], "features": features,
            "samples": torch.cat(samples) if samples else torch.empty(0),
            "gamma": 1.0, "patch_count": patch_index + 1, "spacing": list(module.plan.spacing),
            "batch": reference, "source": self.cases[case], "directory": self.directory,
            "num_classes": self.num_classes, "dice": self.args.dice, "hd": self.args.hd,
            "save_predictions": self.args.save_predictions, "save_orthogonal": self.args.save_orthogonal,
        })
        return {}


class ObservedModel(Model):
    def test_step(self, batch, batch_index):
        return self.observation_predictor(self, batch, batch_index)


class ObserveArguments:
    @staticmethod
    def _hd_percentile(value):
        percentile = float(value)
        if not 0 < percentile <= 100:
            raise argparse.ArgumentTypeError("--hd must be greater than 0 and at most 100")
        return percentile


def _worker(args, meta, custom_class):
    from lightning import Trainer

    directory = args.output
    tempfile.tempdir = str(args._ipc_dir or args.temp_dir)
    settings = _read_json(directory / "input.json")
    if _settings(args, meta) != settings:
        raise ValueError("Dataset or plan changed before worker startup")
    signature, digest = _signature(args._checkpoint), _sha256(args._checkpoint)
    checkpoint = torch.load(args._checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != args._epoch - 1:
        raise ValueError("Checkpoint filename epoch does not match saved Lightning epoch")
    plan = Plan(args.plan)
    saved_plan = checkpoint.get("hyper_parameters", {}).get("plan")
    fields = ("patch_size", "spacing", "mean", "std", "percentile_00_5", "percentile_99_5",
              "pool_strides", "conv_kernel_size", "stem_channel", "max_feature_channel", "n_stages", "dim")
    mismatched = [field for field in fields
                  if saved_plan is None or getattr(saved_plan, field, None) != getattr(plan, field)]
    if mismatched:
        raise ValueError(f"Observation plan differs from checkpoint training plan: {mismatched}")
    del checkpoint, saved_plan
    torch.set_float32_matmul_precision("medium")
    module = ObservedModel.load_from_checkpoint(args._checkpoint, map_location="cpu", strict=True,
                                                weights_only=False, plan=plan)
    head = module.unet.decoder.head
    heads = list(head) if isinstance(head, torch.nn.ModuleList) else [head]
    channels = [getattr(item, "output_channel", None) for item in heads]
    if not channels or any(not isinstance(count, int) or count < 2 for count in channels) or len(set(channels)) != 1:
        raise ValueError(f"Invalid restored output heads: {channels}")
    cases = paired_cases(args.data / image_key, args.data / label_key)
    datamodule = DataModule(args.data / image_key, plan, num_workers=args.loader_workers, direct_test_dir=True)
    trainer = Trainer(accelerator="gpu", devices=[0], logger=False, enable_checkpointing=False, callbacks=[])
    observation = Observation(directory, args, module, cases, channels[0])
    completed = False
    started = time.perf_counter()
    try:
        module.observation_predictor = observation
        sources = {Path(__file__).resolve(), Path(__file__).with_name("observe_cpu.py").resolve(),
                   Path(inspect.getfile(Model)).resolve(), Path(inspect.getfile(DataModule)).resolve(),
                   Path(inspect.getfile(BiMambaBlock)).resolve(),
                   Path(inspect.getfile(BaseObservation)).resolve(),
                   Path(inspect.getfile(full_feature_stats)).resolve(),
                   Path(inspect.getfile(paired_cases)).resolve(),
                   Path(__file__).with_name("__init__.py").resolve()}
        observation.emit({"event": "source", "files_sha256": {str(path): _sha256(path) for path in sorted(sources)}})
        observation.emit({"event": "execution", "temp_dir": str(args.temp_dir),
                          "loader_workers": args.loader_workers, "cpu_workers": args.workers,
                          "max_pending": args.max_pending, "cpu_transport": "disk"})
        trainer.test(model=module, datamodule=datamodule)
        if trainer.interrupted:
            raise KeyboardInterrupt
        observation.finish()
        if _settings(args, meta) != settings or _signature(args._checkpoint) != signature:
            raise ValueError("Inputs changed during observation; result will not be committed")
        _write_json(directory / "complete.json", {
            "schema": SCHEMA, "epoch": args._epoch, "lightning_epoch": args._epoch - 1,
            "checkpoint": signature, "checkpoint_sha256": digest,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "seconds_including_metrics": time.perf_counter() - started, "conditions": list(CONDITIONS),
            "artifacts": {str(path.relative_to(directory)): path.stat().st_size
                          for path in directory.rglob("*")
                          if path.is_file() and path.name not in {"worker.log", "complete.json"}}})
        completed = True
    finally:
        with _ignore_sigint():
            try:
                observation.close(abort=not completed)
            finally:
                module.observation_predictor = None


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
                        default=resolved_path("hidden-tmp"),
                        help="Directory for CPU job files and worker IPC; created if missing (default: ./hidden-tmp).")
    parser.add_argument("--dice", action="store_true")
    parser.add_argument("--hd", type=custom_class._hd_percentile, default=None)
    parser.add_argument("-d", "--devices", type=int, nargs="+", default=None,
                        help="Candidate logical CUDA indices; default: all visible GPUs.")
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
    parser.add_argument("--_ipc-dir", type=resolved_path, help=argparse.SUPPRESS)
    return parser


def _settings(args, meta):
    cases = paired_cases(args.data / image_key, args.data / label_key)
    return {"schema": SCHEMA, "lib": "pretrained_seg", "data": str(args.data), "trained": [str(root) for root in args.trained],
            "plan_sha256": _sha256(args.plan), "experiment": meta.experiment_name,
            "dice": args.dice, "hd": args.hd, "feature_samples": args.feature_samples,
            "observe_metrics": args.observe_metrics,
            "save_predictions": args.save_predictions, "save_orthogonal": args.save_orthogonal,
            "conditions": list(CONDITIONS),
            "cases": {name: {key: _signature(path) for key, path in case.items()}
                      for name, case in cases.items()}}


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


def observe(command, meta, custom_class=ObserveArguments):
    args = _parser(custom_class).parse_args(command)
    groups = set(args.observe_metrics)
    if "all" in groups:
        groups = set(METRIC_GROUPS)
    if "quality" in groups and not args.dice and args.hd is None:
        args.dice = True
    if args.dice or args.hd is not None:
        groups.add("quality")
    args.observe_metrics = [name for name in METRIC_GROUPS if name in groups]
    args.temp_dir.mkdir(parents=True, exist_ok=True)
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
    child_group = None
    epoch_ipc = None
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
                                      "pretrained_seg", "custom"]
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
                    # The monitor owns this root so it can clean disk payloads
                    # after SIGKILL too. Never infer deletion paths from worker logs.
                    epoch_ipc = Path(tempfile.mkdtemp(prefix="pretrained-seg-observe-epoch-", dir=args.temp_dir))
                    environment = _worker_environment(args, gpu, epoch_ipc)
                    worker_command.extend(["--_ipc-dir", str(epoch_ipc)])
                    print(f"Observing epoch {epoch} on {gpu['uuid']}: {attempt}", flush=True)
                    with (attempt / "worker.log").open("x", encoding="utf-8") as log:
                        child = subprocess.Popen(worker_command, env=environment, stdout=log,
                                                 stderr=subprocess.STDOUT,
                                                 start_new_session=os.name != "nt")
                        child_group = child.pid if os.name != "nt" else None
                        next_status = time.monotonic() + 60
                        while child.poll() is None:
                            time.sleep(0.5)
                            if time.monotonic() >= next_status:
                                print(f"Epoch {epoch} running on {gpu['uuid']}; log: {attempt / 'worker.log'}", flush=True)
                                next_status = time.monotonic() + 60
                        code = child.returncode
                        with _ignore_sigint():
                            _stop_epoch(child, child_group)
                        child = None
                        child_group = None
                        _cleanup_epoch_ipc(epoch_ipc)
                        epoch_ipc = None
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
        with _ignore_sigint():
            try:
                if child is not None:
                    # Do not skip descendants just because their leader exited.
                    _stop_epoch(child, child_group if child_group is not None else child.pid)
                _cleanup_epoch_ipc(epoch_ipc)
            finally:
                lock.close()
