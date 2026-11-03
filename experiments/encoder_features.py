"""Extract deterministic, unmasked encoder stage features from image patches."""

import contextlib
import csv
import hashlib
import json
import sys
import time
from itertools import groupby

import torch

from experiments.argument_adaptor import ArgumentAdaptor
from experiments.config import image_key
from experiments.plan import Plan
from experiments.preprocess import load_transformd, planned_transformd
from experiments.trainer import dl_pretrained_unet
from experiments.utils.fsutils import resolved_path


class EncoderFeatureInferencer(ArgumentAdaptor):
    def get_argument_parser(self):
        parser = super().get_argument_parser()
        parser.add_argument("patch_csv", type=resolved_path,
                            help="UTF-8 patch CSV, or one image with --patch-start/--patch-end.")
        parser.add_argument("save_path", type=resolved_path)
        parser.add_argument("plan_path", type=resolved_path)
        parser.add_argument("-c", "--ckpt", required=True, type=resolved_path)
        parser.add_argument("--patch-start", type=int, nargs=3)
        parser.add_argument("--patch-end", type=int, nargs=3)
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument(
            "--save-dtype", choices=("float32", "float16", "bfloat16"), default="float32"
        )
        return parser

    @staticmethod
    def _patch_name(start, end):
        return "patch_" + "_".join(map(str, start)) + "__" + "_".join(map(str, end))

    def _read_patches(self):
        args = self.args
        if args.patch_start is not None or args.patch_end is not None:
            if args.patch_start is None or args.patch_end is None:
                raise ValueError("--patch-start and --patch-end must be provided together.")
            name = args.patch_csv.name
            case_id = name[:-7] if name.endswith(".nii.gz") else args.patch_csv.stem
            rows = [{"case_id": case_id, "case_path": str(args.patch_csv),
                     "patch_id": self._patch_name(args.patch_start, args.patch_end),
                     **dict(zip((f"start_{axis}" for axis in "hwd"), args.patch_start, strict=True)),
                     **dict(zip((f"end_{axis}" for axis in "hwd"), args.patch_end, strict=True))}]
        else:
            with args.patch_csv.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                required = {"case_id", "case_path", "patch_id",
                            "start_h", "start_w", "start_d", "end_h", "end_w", "end_d"}
                if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                    raise ValueError(f"CSV requires columns: {sorted(required)}")
                rows = list(reader)
        patches, cases, seen = [], {}, set()
        for line, row in enumerate(rows, start=2):
            case_id, patch_id = row["case_id"], row["patch_id"]
            if (not case_id or case_id in {".", ".."}
                    or any(char in case_id for char in '/\\:*?"<>|') or case_id.endswith((".", " "))):
                raise ValueError(f"Invalid case_id at row {line}: {case_id!r}")
            if not patch_id or not row["case_path"]:
                raise ValueError(f"Missing patch_id or case_path at row {line}")
            path = resolved_path(row["case_path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            if case_id in cases and cases[case_id] != path:
                raise ValueError(f"Different case_path values for case_id {case_id!r}")
            cases[case_id] = path
            start = tuple(int(row[f"start_{axis}"]) for axis in "hwd")
            end = tuple(int(row[f"end_{axis}"]) for axis in "hwd")
            if any(s < 0 or e <= s for s, e in zip(start, end, strict=True)):
                raise ValueError(f"Require 0 <= start < end on every axis at row {line}")
            key = (case_id, start, end)
            if key in seen:
                raise ValueError(f"Duplicate patch coordinates at row {line}: {key}")
            seen.add(key)
            patches.append((case_id, path, patch_id, start, end))
        if not patches:
            raise ValueError("Patch CSV contains no patches.")
        return sorted(patches, key=lambda patch: patch[0])

    @staticmethod
    def _file_signature(path):
        stat = path.stat()
        return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def _save_feature(self, group, name, feature):
        dtype = getattr(torch, self.args.save_dtype)
        tensor = feature.detach().squeeze(0).to(device="cpu", dtype=dtype).contiguous()
        array = (
            tensor.view(torch.uint16).numpy()
            if dtype == torch.bfloat16
            else tensor.numpy()
        )
        chunks = (min(array.shape[0], 8), *(min(n, 32) for n in array.shape[1:]))
        dataset = group.create_array(name, data=array, chunks=chunks)
        dataset.attrs.update({"logical_dtype": self.args.save_dtype, "axes": ["C", "H", "W", "D"]})

    def __call__(self):
        try:
            import zarr
        except ImportError as exc:
            raise RuntimeError("Encoder feature export requires zarr>=3 in the container.") from exc

        args = self.args
        for path in (args.patch_csv, args.plan_path, args.ckpt):
            if not path.is_file():
                raise FileNotFoundError(path)
        patches = self._read_patches()
        common = {"checkpoint": self._file_signature(args.ckpt),
                  "plan_sha256": hashlib.sha256(args.plan_path.read_bytes()).hexdigest()}
        provenances = {}
        # Check all output collisions before loading the model or writing anything.
        for case_id, case_patches in groupby(patches, key=lambda patch: patch[0]):
            case_patches = list(case_patches)
            destination = args.save_path / f"{case_id}-features.zarr"
            provenance = {"case": self._file_signature(case_patches[0][1]), **common}
            provenances[case_id] = provenance
            if destination.exists():
                existing = zarr.open_group(str(destination), mode="r")
                if existing.attrs.get("provenance") != provenance:
                    raise ValueError(f"Different image, checkpoint or plan at {destination}; use another output directory.")
                for _, _, _, start, end in case_patches:
                    patch_name = self._patch_name(start, end)
                    if patch_name in existing:
                        raise FileExistsError(f"Patch already exists: {destination}/{patch_name}")

        # Third-party diagnostics go to stderr; stdout contains only completion JSON.
        with contextlib.redirect_stdout(sys.stderr):
            plan = Plan(args.plan_path)
            unet = dl_pretrained_unet(args.ckpt).eval().to(args.device)
            load = load_transformd([image_key])
            preprocess = planned_transformd(plan, (image_key,))
        for case_id, case_patches in groupby(patches, key=lambda patch: patch[0]):
            case_patches = list(case_patches)
            case_path = case_patches[0][1]
            with contextlib.redirect_stdout(sys.stderr):
                image = preprocess(load({image_key: str(case_path)}))[image_key]
            spatial_shape = tuple(image.shape[1:])
            if image.ndim != 4:
                raise ValueError(f"Expected (C,H,W,D), got {tuple(image.shape)}")
            if image.shape[0] != unet.input_channel:
                raise ValueError(f"Expected {unet.input_channel} input channels, got {image.shape[0]}.")
            for _, _, _, start, end in case_patches:
                if any(e > n for e, n in zip(end, spatial_shape, strict=True)):
                    raise ValueError(f"Patch {start}:{end} is outside preprocessed image shape {spatial_shape}.")
            affine = image.affine.cpu().tolist()
            destination = args.save_path / f"{case_id}-features.zarr"
            args.save_path.mkdir(parents=True, exist_ok=True)
            root = zarr.open_group(str(destination), mode="a", zarr_format=2)
            root.attrs["provenance"] = provenances[case_id]
            for _, _, patch_id, start, end in case_patches:
                started = time.perf_counter()
                slices = tuple(slice(s, e) for s, e in zip(start, end, strict=True))
                patch = image[(slice(None), *slices)].as_tensor().unsqueeze(0)
                patch = patch.to(device=args.device, dtype=torch.float32)
                group = root.create_group(self._patch_name(start, end), overwrite=False)
                group.attrs.update({
                    "complete": False,
                    "patch_start": list(start), "patch_end": list(end),
                    "preprocessed_shape": list(spatial_shape),
                    "coordinate_space": "plan_preprocessed_HWD_end_exclusive",
                    "preprocessed_affine": affine,
                    "logical_dtype": args.save_dtype, "masked": False,
                })
                # Save each stage immediately, without retaining all GPU feature maps.
                with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                    feature = unet.encoder.stem(patch)
                    self._save_feature(group, "stem", feature)
                    for index, stage in enumerate(unet.encoder.stages):
                        feature = stage(feature)
                        self._save_feature(group, f"stage_{index}", feature)
                group.attrs["complete"] = True
                record = {
                    "complete": True, "case_id": case_id, "patch_id": patch_id,
                    "patch_start_hwd": list(start), "patch_end_hwd": list(end),
                    "preprocessed_shape": list(spatial_shape), "preprocessed_affine": affine,
                    "type": args.save_dtype, "elapsed_time": time.perf_counter() - started,
                }
                print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
                del patch, feature
            del image

