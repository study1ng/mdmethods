"""Extract deterministic, unmasked encoder stage features from one image patch."""

import hashlib

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
        parser.add_argument("case_path", type=resolved_path)
        parser.add_argument("save_path", type=resolved_path)
        parser.add_argument("plan_path", type=resolved_path)
        parser.add_argument("-c", "--ckpt", required=True, type=resolved_path)
        parser.add_argument("--patch-start", required=True, type=int, nargs=3)
        parser.add_argument("--patch-end", required=True, type=int, nargs=3)
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument(
            "--save-dtype", choices=("float32", "float16", "bfloat16"), default="float32"
        )
        return parser

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
        for path in (args.case_path, args.plan_path, args.ckpt):
            if not path.is_file():
                raise FileNotFoundError(path)
        start, end = tuple(args.patch_start), tuple(args.patch_end)
        if any(s < 0 or e <= s for s, e in zip(start, end, strict=True)):
            raise ValueError("Require 0 <= patch-start < patch-end on every axis.")

        case_name = args.case_path.name
        case_id = case_name[:-7] if case_name.endswith(".nii.gz") else args.case_path.stem
        destination = args.save_path / f"{case_id}-features.zarr"
        patch_name = "patch_" + "_".join(map(str, start)) + "__" + "_".join(map(str, end))
        provenance = {
            "case": self._file_signature(args.case_path),
            "checkpoint": self._file_signature(args.ckpt),
            "plan_sha256": hashlib.sha256(args.plan_path.read_bytes()).hexdigest(),
        }
        # Inspect existing output without creating anything, before expensive inference.
        if destination.exists():
            existing = zarr.open_group(str(destination), mode="r")
            if existing.attrs.get("provenance") != provenance:
                raise ValueError(f"Different image, checkpoint or plan at {destination}; use another output directory.")
            if patch_name in existing:
                raise FileExistsError(f"Patch already exists: {destination}/{patch_name}")

        plan = Plan(args.plan_path)
        image = load_transformd([image_key])({image_key: str(args.case_path)})
        image = planned_transformd(plan, (image_key,))(image)[image_key]
        spatial_shape = tuple(image.shape[1:])
        if image.ndim != 4 or any(e > n for e, n in zip(end, spatial_shape, strict=True)):
            raise ValueError(f"Patch {start}:{end} is outside preprocessed image shape {spatial_shape}.")
        slices = tuple(slice(s, e) for s, e in zip(start, end, strict=True))
        patch = image[(slice(None), *slices)].as_tensor().unsqueeze(0)

        unet = dl_pretrained_unet(args.ckpt).eval().to(args.device)
        if patch.shape[1] != unet.input_channel:
            raise ValueError(f"Expected {unet.input_channel} input channels, got {patch.shape[1]}.")
        patch = patch.to(device=args.device, dtype=torch.float32)
        print(f"Saving encoder features to {destination / patch_name}")
        args.save_path.mkdir(parents=True, exist_ok=True)
        root = zarr.open_group(str(destination), mode="a", zarr_format=2)
        root.attrs["provenance"] = provenance
        group = root.create_group(patch_name, overwrite=False)
        group.attrs.update({
            "complete": False,
            "patch_start": list(start),
            "patch_end": list(end),
            "preprocessed_shape": list(spatial_shape),
            "coordinate_space": "plan_preprocessed_HWD_end_exclusive",
            "preprocessed_affine": image.affine.cpu().tolist(),
            "logical_dtype": args.save_dtype,
            "masked": False,
        })
        # Save each stage immediately, without retaining all GPU feature maps.
        with torch.inference_mode():
            feature = unet.encoder.stem(patch)
            self._save_feature(group, "stem", feature)
            for index, stage in enumerate(unet.encoder.stages):
                feature = stage(feature)
                self._save_feature(group, f"stage_{index}", feature)
        group.attrs["complete"] = True

