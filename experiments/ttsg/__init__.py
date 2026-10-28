"""TotalSegmentator inference and evaluation through the experiment CLI."""

import argparse
import json
from importlib.metadata import version

import nibabel as nib
import numpy as np

from experiments import ArgumentAdaptor
from experiments.config import filekey, image_key, label_key
from experiments.munet.evaluation import paired_cases, write_metrics
from experiments.utils import nowstring, resolved_path


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return value


def _hd_percentile(value):
    value = float(value)
    if not 0 < value <= 100:
        raise argparse.ArgumentTypeError("--hd must be greater than 0 and at most 100")
    return value


def _save_labels(values, source, path):
    header = source.header.copy()
    header.set_data_dtype(np.int32)
    image = nib.Nifti1Image(values, source.affine, header)
    for name in ("qform", "sform"):
        affine, code = getattr(source, f"get_{name}")(coded=True)
        getattr(image, f"set_{name}")(affine, int(code))
    if path.exists():
        raise FileExistsError(f"Output file already exists: {path}")
    nib.save(image, path)


class TTSGInferencer(ArgumentAdaptor):
    def get_argument_parser(self):
        parser = super().get_argument_parser()
        parser.add_argument("data", type=resolved_path)
        parser.add_argument("save_path", type=resolved_path)
        parser.add_argument("--task", default="total")
        parser.add_argument("--roi_subset", "--roi-subset", nargs="+")
        parser.add_argument("--label-map", type=resolved_path,
                            help="JSON object mapping organ names to reference label IDs")
        parser.add_argument("-d", "--devices", type=int, nargs="+", default=[0])
        parser.add_argument("--cpu", action="store_true")
        parser.add_argument("--fast", action="store_true")
        return parser

    def parse_args(self, args):
        super().parse_args(args)
        if len(self.args.devices) != 1 or self.args.devices[0] < 0:
            raise ValueError("TotalSegmentator requires exactly one nonnegative GPU ID")
        if self.args.roi_subset and self.args.task not in {"total", "total_mr"}:
            raise ValueError("--roi_subset is supported only for total and total_mr")
        self.save_path = self.args.save_path / self.meta.lib
        if self.meta.experiment_name is not None:
            self.save_path /= self.meta.experiment_name
        self.save_path /= nowstring()
        if self.save_path.exists():
            raise FileExistsError(f"Output directory already exists: {self.save_path}")

    def _cases(self):
        if not self.args.data.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {self.args.data}")
        cases = {}
        for path in sorted(self.args.data.glob("*.nii.gz")):
            if not path.is_file():
                continue
            name = filekey(path)
            if name in cases:
                raise ValueError(f"Duplicate case ID: {name}")
            cases[name] = {"input_image": path}
        if not cases:
            raise ValueError(f"No .nii.gz files in {self.args.data}")
        return cases

    def _labels(self):
        from totalsegmentator.map_to_binary import class_map

        if self.args.task not in class_map:
            raise ValueError(f"Unknown label set for task: {self.args.task}")
        task_labels = class_map[self.args.task]
        names = self.args.roi_subset
        if names and (len(names) != len(set(names)) or set(names) - set(task_labels.values())):
            raise ValueError("--roi_subset contains duplicate or unknown organ names")
        selected = [(int(key), name) for key, name in task_labels.items()
                    if names is None or name in names]
        selected.sort()
        mapping = None
        if self.args.label_map is not None:
            with self.args.label_map.open(encoding="utf-8") as stream:
                mapping = json.load(stream)
            if not isinstance(mapping, dict) or any(
                type(value) is not int or value < 1 for value in mapping.values()
            ):
                raise ValueError("--label-map must map organ names to positive integer IDs")
            if set(mapping) != {name for _, name in selected}:
                raise ValueError("--label-map must contain exactly the selected organ names")
            if len(set(mapping.values())) != len(mapping):
                raise ValueError("Reference label IDs must be unique")
        if not selected:
            raise ValueError("No foreground classes selected")
        return [
            {"output_id": index, "name": name, "totalsegmentator_id": key,
             "reference_id": mapping[name] if mapping is not None else key}
            for index, (key, name) in enumerate(selected, 1)
        ]

    @staticmethod
    def _remap(image, labels, source_key):
        values = np.asanyarray(image.dataobj)
        if (values.ndim != 3 or not np.isfinite(values).all()
                or (values < 0).any() or not np.equal(values, np.floor(values)).all()):
            raise ValueError("Expected a 3D label map containing nonnegative integers")
        result = np.zeros(values.shape, dtype=np.int32)
        for label in labels:
            result[values == label[source_key]] = label["output_id"]
        return result

    def __call__(self):
        # Import lazily so module help does not require the optional dependency.
        try:
            from totalsegmentator.python_api import totalsegmentator
        except ModuleNotFoundError as exc:
            if exc.name == "totalsegmentator":
                raise RuntimeError("Install TotalSegmentator manually in the project container") from exc
            raise
        cases = self._cases()
        labels = self._labels()
        evaluating = isinstance(self, TTSGValidator)
        # Validate references before downloading weights or creating output.
        if evaluating:
            for case in cases.values():
                self._remap(nib.load(case["reference_label"]), labels, "reference_id")
        print(f"Inference and metrics output: {self.save_path}")
        self.save_path.mkdir(parents=True, exist_ok=False)
        manifest = {
            "totalsegmentator_version": version("TotalSegmentator"),
            "task": self.args.task, "roi_subset": self.args.roi_subset,
            "fast": self.args.fast, "labels": labels,
        }
        with (self.save_path / "labels.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
        if evaluating:
            (self.save_path / "reference").mkdir()
        for name, case in cases.items():
            prediction = totalsegmentator(
                case["input_image"], output=None, ml=True,
                task=self.args.task, roi_subset=self.args.roi_subset,
                device="cpu" if self.args.cpu else f"gpu:{self.args.devices[0]}",
                fast=self.args.fast,
            )
            _save_labels(
                self._remap(prediction, labels, "totalsegmentator_id"), prediction,
                self.save_path / f"test_0_{name}_out.nii.gz",
            )
            if evaluating:
                reference = nib.load(case["reference_label"])
                reference_path = self.save_path / "reference" / f"{name}.nii.gz"
                _save_labels(self._remap(reference, labels, "reference_id"), reference, reference_path)
                case["reference_label"] = reference_path
        if evaluating:
            write_metrics(
                cases, self.save_path, epoch=0, num_classes=len(labels) + 1,
                dice=self.args.dice, hd=self.args.hd, workers=self.args.workers,
            )


class TTSGValidator(TTSGInferencer):
    def get_argument_parser(self):
        parser = super().get_argument_parser()
        parser.add_argument("--dice", action="store_true")
        parser.add_argument("--hd", type=_hd_percentile)
        parser.add_argument("-w", "--workers", type=_positive_int, default=4)
        return parser

    def _cases(self):
        return paired_cases(self.args.data / image_key, self.args.data / label_key)


def inference(args, meta):
    TTSGInferencer(args, meta)()


def custom(args, meta):
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["val"])
    if args and args[0] == "val":
        TTSGValidator(args[1:], meta)()
    else:
        parser.parse_args(args)
