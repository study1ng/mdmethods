"""Evaluate labels on an orthogonal grid close to the input image on CPU."""

import csv
import math
import re
from pathlib import Path
from pprint import pprint
from itertools import product

import nibabel as nib
import numpy as np
import torch
from monai.metrics import DiceMetric, compute_hausdorff_distance
from monai.data import MetaTensor
from monai.transforms import ResampleToMatch

from experiments.config import filekey


def paired_cases(image_dir: Path, label_dir: Path):
    def index(directory):
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {directory}")
        cases = {}
        for path in sorted(directory.glob("*.nii.gz")):
            if not path.is_file():
                continue
            name = filekey(path)
            if name in cases:
                raise ValueError(f"Duplicate case ID {name!r} in {directory}")
            cases[name] = path
        if not cases:
            raise ValueError(f"No .nii.gz files in {directory}")
        return cases

    images, labels = index(image_dir), index(label_dir)
    if images.keys() != labels.keys():
        raise ValueError(
            f"Image/label case IDs differ: missing labels={sorted(images.keys() - labels.keys())}, "
            f"missing images={sorted(labels.keys() - images.keys())}"
        )
    return {
        name: {"input_image": images[name], "reference_label": labels[name]}
        for name in images
    }


def _load_labels(image, num_classes=None):
    values = np.asanyarray(image.dataobj)
    if values.ndim != 3:
        raise ValueError(f"Expected a 3D label map: {image.get_filename()}")
    if (
        not np.isfinite(values).all()
        or not np.equal(values, np.floor(values)).all()
        or values.min() < 0
        or (num_classes is not None and values.max() >= num_classes)
    ):
        raise ValueError(
            f"Labels must be nonnegative integers"
            f"{f' below {num_classes}' if num_classes is not None else ''}: "
            f"{image.get_filename()}"
        )
    return torch.from_numpy(values.astype(np.int64))[None, None]


def _mean(values):
    values = [value for value in values if not math.isnan(value)]
    return math.fsum(values) / len(values) if values else float("nan")


def _prediction_paths(cases, save_path, epoch):
    if not save_path.is_dir():
        raise FileNotFoundError(f"Prediction directory does not exist: {save_path}")
    if epoch is not None:
        return {
            name: save_path / f"test_{epoch}_{name}_out.nii.gz" for name in cases
        }
    predictions = {}
    for path in sorted(save_path.glob("test_*_out.nii.gz")):
        match = re.fullmatch(r"test_\d+_(.+)_out\.nii\.gz", path.name)
        if match is None or not path.is_file():
            continue
        name = match.group(1)
        if name in predictions:
            raise ValueError(f"Multiple predictions for case {name}: {predictions[name]}, {path}")
        predictions[name] = path
    if predictions.keys() != cases.keys():
        raise ValueError(
            f"Prediction case IDs differ: missing predictions={sorted(cases.keys() - predictions.keys())}, "
            f"unexpected predictions={sorted(predictions.keys() - cases.keys())}"
        )
    return predictions


def _print_metadata(case, prediction_path):
    paths = {
        "input_image": case["input_image"],
        "prediction_label": prediction_path,
        "reference_label": case["reference_label"],
    }
    for key, path in paths.items():
        try:
            image = nib.load(path)
            qform, qform_code = image.get_qform(coded=True)
            sform, sform_code = image.get_sform(coded=True)
            metadata = {
                "path": str(path), "shape": image.shape, "affine": image.affine,
                "spacing": image.header.get_zooms(),
                "units": image.header.get_xyzt_units(),
                "qform": qform, "qform_code": int(qform_code),
                "sform": sform, "sform_code": int(sform_code),
            }
        except Exception as exc:
            metadata = {"path": str(path), "error": str(exc)}
        print(f"{key}: ", end="")
        pprint(metadata, sort_dicts=False)


def _affine_mm(image):
    units = image.header.get_xyzt_units()[0]
    factors = {"unknown": 1, "mm": 1, "meter": 1000, "micron": 0.001}
    if units not in factors:
        raise ValueError(f"Unsupported spatial units: {units}")
    affine = image.affine.copy()
    affine[:3, :] *= factors[units]
    if not np.isfinite(affine).all() or np.linalg.matrix_rank(affine[:3, :3]) != 3:
        raise ValueError(f"Invalid affine: {image.get_filename()}")
    return affine


def _evaluation_grid(shape, affine):
    """Keep voxel sizes and center; remove shear with the closest orthogonal axes.

    Bound voxel edges, not just centers, so the full input field of view fits.
    Reflections are preserved rather than forcing a right-handed orientation.
    """
    shape = np.asarray(shape, dtype=np.int64)
    if shape.shape != (3,) or (shape <= 0).any():
        raise ValueError("Expected a nonempty 3D input image")
    axes = affine[:3, :3]
    spacing = np.linalg.norm(axes, axis=0)
    directions = axes / spacing
    if np.allclose(directions.T @ directions, np.eye(3), rtol=0, atol=1e-10):
        return tuple(int(size) for size in shape), affine.copy(), spacing
    left, _, right = np.linalg.svd(directions)
    orthogonal = left @ right
    center = axes @ ((shape - 1) / 2) + affine[:3, 3]
    corners = np.asarray(list(product(*[(-0.5, size - 0.5) for size in shape])))
    world_corners = corners @ axes.T + affine[:3, 3]
    local_corners = (world_corners - center) @ orthogonal
    extent = 2 * np.max(np.abs(local_corners), axis=0)
    # Suppress floating-point noise at integral voxel counts, not real extent.
    target_shape = np.maximum(1, np.ceil(extent / spacing - 1e-10)).astype(np.int64)
    target_affine = np.eye(4)
    target_affine[:3, :3] = orthogonal * spacing
    target_affine[:3, 3] = center - target_affine[:3, :3] @ ((target_shape - 1) / 2)
    return tuple(int(size) for size in target_shape), target_affine, spacing


def _save_orthogonal(value, affine, path, *, is_label):
    if path.exists():
        raise FileExistsError(f"Output file already exists: {path}")
    array = value.detach().cpu().numpy()
    dtype = np.int64 if is_label else np.float32
    image = nib.Nifti1Image(array, affine, dtype=dtype)
    image.header.set_xyzt_units("mm")
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, path)


def _case_labels(case, prediction_path, num_classes, hd, *, orthogonal_dir=None):
    input_image = nib.load(case["input_image"])
    if len(input_image.shape) != 3:
        raise ValueError("Expected a 3D input image")
    input_affine = _affine_mm(input_image)
    target_shape, target_affine, spacing = _evaluation_grid(input_image.shape, input_affine)
    prediction_image = nib.load(prediction_path)
    reference_image = nib.load(case["reference_label"])
    # Only shape and affine are needed from the target, not its intensity data.
    target = MetaTensor(
        torch.empty((1, *target_shape), dtype=torch.uint8), affine=target_affine
    )
    resampler = ResampleToMatch(mode="nearest", padding_mode="zeros")

    def aligned_labels(image):
        labels = _load_labels(image, num_classes)
        source = MetaTensor(labels[0], affine=_affine_mm(image))
        aligned = resampler(source, target)
        return aligned.as_tensor().to(torch.int64)[None]

    prediction = aligned_labels(prediction_image)
    reference = aligned_labels(reference_image)
    if orthogonal_dir is not None:
        input_values = input_image.get_fdata(dtype=np.float32)
        if not np.isfinite(input_values).all():
            raise ValueError("Input image contains nonfinite intensities")
        input_tensor = MetaTensor(torch.from_numpy(input_values)[None], affine=input_affine)
        aligned_input = ResampleToMatch(mode="bilinear", padding_mode="zeros")(
            input_tensor, target
        )
        prefix = prediction_path.name.removesuffix("_out.nii.gz")
        for suffix, value, is_label in (
            ("image", aligned_input.as_tensor()[0], False),
            ("gt", reference[0, 0], True),
            ("out", prediction[0, 0], True),
        ):
            _save_orthogonal(
                value, target_affine, orthogonal_dir / f"{prefix}_{suffix}.nii.gz",
                is_label=is_label,
            )
    return prediction, reference, spacing


def write_metrics(cases, save_path: Path, *, epoch=None, num_classes=None, dice, hd):
    if num_classes is not None and num_classes < 2:
        raise ValueError("Evaluation requires at least one foreground class")
    metrics_dir = save_path / "metrics"
    orthogonal_dir = save_path / "orthogonal"
    for directory in (metrics_dir, orthogonal_dir):
        if directory.exists():
            raise FileExistsError(f"Output directory already exists: {directory}")
    predictions = _prediction_paths(cases, save_path, epoch)
    organs = set(range(1, num_classes)) if num_classes is not None else set()
    if num_classes is None:
        # Discover a dataset-wide vocabulary without a model or plan.
        # Reload one case at a time below to keep memory bounded.
        for name, case in cases.items():
            try:
                prediction, reference, _ = _case_labels(case, predictions[name], None, hd)
                for labels in (prediction, reference):
                    organs.update(int(value) for value in torch.unique(labels).tolist() if value != 0)
            except Exception:
                _print_metadata(case, predictions[name])
                raise
        if not organs:
            raise ValueError("Evaluation requires at least one foreground label in prediction or reference")
    organs = sorted(organs)
    columns = (["dice"] if dice else []) + ([f"hd{hd:g}_mm"] if hd is not None else [])
    dice_metric = DiceMetric(
        include_background=True, reduction="none", ignore_empty=True,
    ) if dice else None
    rows = []
    orthogonal_dir.mkdir(parents=True, exist_ok=False)
    for name, case in cases.items():
        try:
            prediction, reference, spacing = _case_labels(
                case, predictions[name], num_classes, hd, orthogonal_dir=orthogonal_dir
            )
            for organ in organs:
                row = {"case": name, "organ": organ}
                # One binary organ at a time also supports noncontiguous label IDs.
                predicted_organ, reference_organ = prediction == organ, reference == organ
                if dice:
                    row["dice"] = dice_metric(predicted_organ, reference_organ)[0, 0].item()
                    dice_metric.reset()
                if hd is not None:
                    row[columns[-1]] = compute_hausdorff_distance(
                        predicted_organ, reference_organ,
                        include_background=True, percentile=hd, directed=False,
                        spacing=spacing.tolist(),
                    )[0, 0].item()
                rows.append(row)
        except Exception:
            _print_metadata(case, predictions[name])
            raise

    def averaged(group):
        groups = {}
        for row in rows:
            groups.setdefault(row[group], []).append(row)
        return [
            {group: key, **{
                column: _mean([row[column] for row in members])
                for column in columns
            }}
            for key, members in groups.items()
        ]

    metrics_dir.mkdir(parents=True, exist_ok=False)
    for filename, fields, values in (
        ("full.csv", ["case", "organ", *columns], rows),
        ("case.csv", ["case", *columns], averaged("case")),
        ("organ.csv", ["organ", *columns], averaged("organ")),
    ):
        with (metrics_dir / filename).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
