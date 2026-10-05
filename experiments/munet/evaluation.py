"""Evaluate saved inference labels in their original voxel grid on CPU."""

import csv
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.metrics import DiceMetric, compute_hausdorff_distance

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
    return {name: labels[name] for name in images}


def _load_labels(image, num_classes):
    values = np.asanyarray(image.dataobj)
    if values.ndim != 3:
        raise ValueError(f"Expected a 3D label map: {image.get_filename()}")
    if (
        not np.isfinite(values).all()
        or not np.equal(values, np.floor(values)).all()
        or values.min() < 0
        or values.max() >= num_classes
    ):
        raise ValueError(
            f"Labels must be integers in [0, {num_classes - 1}]: {image.get_filename()}"
        )
    return torch.from_numpy(values.astype(np.int64))[None, None]


def _mean(values):
    values = [value for value in values if not math.isnan(value)]
    return math.fsum(values) / len(values) if values else float("nan")


def write_metrics(cases, save_path: Path, *, epoch, num_classes, dice, hd):
    if num_classes < 2:
        raise ValueError("Evaluation requires at least one foreground class")
    metrics_dir = save_path / "metrics"
    if metrics_dir.exists():
        raise FileExistsError(f"Metrics directory already exists: {metrics_dir}")
    columns = (["dice"] if dice else []) + ([f"hd{hd:g}_mm"] if hd is not None else [])
    dice_metric = DiceMetric(
        include_background=False, reduction="none", ignore_empty=True,
        num_classes=num_classes,
    ) if dice else None
    rows = []
    for name, label_path in cases.items():
        prediction_path = save_path / f"test_{epoch}_{name}_out.nii.gz"
        prediction_image = nib.load(prediction_path)
        reference_image = nib.load(label_path)
        if prediction_image.shape != reference_image.shape or not np.allclose(
            prediction_image.affine, reference_image.affine, rtol=1e-5, atol=1e-4
        ):
            raise ValueError(f"Prediction and label voxel grids differ for case {name}")
        prediction = _load_labels(prediction_image, num_classes)
        reference = _load_labels(reference_image, num_classes)
        # Euclidean spacing assumes orthogonal voxel axes; reject sheared grids.
        axes = reference_image.affine[:3, :3]
        spacing = np.linalg.norm(axes, axis=0)
        if hd is not None:
            if not np.isfinite(spacing).all() or (spacing <= 0).any():
                raise ValueError(f"Invalid voxel spacing for case {name}")
            directions = axes / spacing
            if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-4):
                raise ValueError(f"HD requires an orthogonal voxel grid: {name}")
            units = reference_image.header.get_xyzt_units()[0]
            if units not in {"unknown", "mm", "meter", "micron"}:
                raise ValueError(f"Unsupported spatial units for case {name}: {units}")
            spacing = spacing * {"unknown": 1, "mm": 1, "meter": 1000, "micron": 0.001}[units]
        dice_values = dice_metric(prediction, reference)[0] if dice else None
        if dice:
            dice_metric.reset()
        for organ in range(1, num_classes):
            row = {"case": name, "organ": organ}
            if dice:
                row["dice"] = dice_values[organ - 1].item()
            if hd is not None:
                # One binary organ at a time avoids a full-volume one-hot allocation.
                row[columns[-1]] = compute_hausdorff_distance(
                    prediction == organ, reference == organ,
                    include_background=True, percentile=hd, directed=False,
                    spacing=spacing.tolist(),
                )[0, 0].item()
            rows.append(row)

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
