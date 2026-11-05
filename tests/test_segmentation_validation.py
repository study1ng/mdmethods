"""Run manually in the container; synthetic CPU inputs, no training or CUDA."""

import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
from monai.data import list_data_collate
from monai.transforms import Compose, CropForegroundd, Spacingd, SpatialPadd

from experiments.callbacks import LogCallback
from experiments.config import image_key, label_key
from experiments.munet.evaluation import _evaluate_case
from experiments.preprocess import load_transformd
from experiments.segmentation_validation import SegmentationValidation


class SegmentationValidationTests(unittest.TestCase):
    def test_validation_matches_saved_inference_after_spatial_transforms(self):
        with TemporaryDirectory(prefix="segmentation-validation-") as temporary:
            root = Path(temporary)
            affine = np.array([[1.2, 0.15, 0, 3.1], [0, 1.5, 0, -2.3],
                               [0, 0, 2.1, 1.7], [0, 0, 0, 1]])
            image = np.full((12, 13, 14), -1000, dtype=np.float32)
            image[2:10, 2:11, 2:12] = 10
            label = np.zeros(image.shape, dtype=np.int16)
            label[3:6, 3:7, 4:9] = 1
            label[7:9, 7:10, 6:10] = 2
            source = {"input_image": root / "image.nii.gz",
                      "reference_label": root / "label.nii.gz"}
            for key, data in (("input_image", image), ("reference_label", label)):
                nifti = nib.Nifti1Image(data, affine)
                nifti.header.set_xyzt_units("mm")
                nib.save(nifti, source[key])
            keys = [image_key, label_key]
            transform = Compose([
                load_transformd(keys),
                CropForegroundd(keys, source_key=image_key,
                                select_fn=lambda value: value > -900),
                Spacingd(keys, pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
                SpatialPadd(keys, spatial_size=(16, 16, 20)),
            ])
            batch = list_data_collate([transform({
                image_key: source["input_image"], label_key: source["reference_label"],
                "name": "synthetic",
            })])
            prediction = batch[label_key].clone()
            prediction[prediction == 2] = 0  # Include a completely missed organ.
            metric = SegmentationValidation()
            metric.update(prediction, batch, 4)  # Class 3 has empty GT.
            rows = next(iter(metric.cases.values()))
            self.assertEqual(rows[1]["dice"], 0)
            self.assertTrue(math.isnan(rows[2]["dice"]))
            output = root / "predictions"
            LogCallback(output)._process_action(
                "test", "out", batch, 0, ("image", prediction), True,
            )
            expected = _evaluate_case(
                "synthetic", source, output / "test_0_synthetic_out.nii.gz",
                4, range(1, 4), True, None, None,
            )
            for actual_row, expected_row in zip(rows, expected, strict=True):
                self.assertEqual(actual_row["organ"], expected_row["organ"])
                if math.isnan(expected_row["dice"]):
                    self.assertTrue(math.isnan(actual_row["dice"]))
                else:
                    self.assertAlmostEqual(actual_row["dice"], expected_row["dice"], places=7)
            self.assertEqual(prediction.device.type, "cpu")

    def test_organ_macro_mean_and_distributed_case_deduplication(self):
        case_a = [{"case": "a", "organ": 1, "dice": 1.0},
                  {"case": "a", "organ": 2, "dice": float("nan")}]
        case_b = [{"case": "b", "organ": 1, "dice": 1.0},
                  {"case": "b", "organ": 2, "dice": 0.0}]
        metric = SegmentationValidation()
        metric.cases = {"a": case_a, "b": case_b}
        with patch("torch.distributed.is_initialized", return_value=False):
            self.assertEqual(metric.aggregate(), 0.5)

        def gather(output, local):
            output[:] = [local, {"a": case_a, "b": case_b}]

        metric.cases = {"a": case_a}
        with patch("torch.distributed.is_available", return_value=True), \
             patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_world_size", return_value=2), \
             patch("torch.distributed.all_gather_object", side_effect=gather):
            self.assertEqual(metric.aggregate(), 0.5)
        metric.reset()
        self.assertEqual(metric.cases, {})


if __name__ == "__main__":
    unittest.main()
