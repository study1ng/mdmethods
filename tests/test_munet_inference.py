"""CPU restoration checks; run manually inside the project container.

Compiled dependencies must be importable, but the Mamba operation is replaced.
No dataset, trainer, logger, or CUDA execution is used.
"""

import importlib
import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import lightning
import torch
from torch import nn

from experiments.config import image_key
from experiments.munet import MUNetInferencer, inference
from experiments.munet.model import MUNetTrainingModule
from experiments.nets.builder import Builder
from experiments.plan import Plan


class CPUMamba(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, value):
        return value * self.scale


class MUNetInferenceTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="munet-inference-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps({
            "foreground_intensity_properties_per_channel": {"0": {
                "max": 1, "min": -1, "std": 1, "mean": 0, "median": 0,
                "percentile_00_5": -1, "percentile_99_5": 1,
            }},
            "configurations": {"3d_fullres": {
                "patch_size": [8, 8, 8], "batch_size": 1,
                "spacing": [1, 1, 1],
                "pool_op_kernel_sizes": [[2, 2, 2], [2, 2, 2]],
                "conv_kernel_sizes": [[3, 3, 3]] * 3,
                "UNet_base_num_features": 4, "unet_max_num_features": 16,
            }},
        }), encoding="utf-8")
        self.plan = Plan(plan_path)
        replacement = patch(
            "experiments.munet.munet_bottleneck.BiMambaBlock", CPUMamba
        )
        replacement.start()
        self.addCleanup(replacement.stop)
        builder = Builder().based_on(
            "nets.plainunet.PlainUNet", n_stages=2, input_channel=1,
            skip_channels=(4, 8, 16), output_channel=2, dim=3,
        ).to_params()
        self.model = MUNetTrainingModule(
            builder, plan=self.plan, overlap_scale=0.0,
            global_positional_encoding_proposition=0.25, gamma=0.3,
        ).eval()
        with torch.no_grad():
            self.model.bottleneck.gamma.fill_(0.7)
            self.model.bottleneck.bottleneck.scale.fill_(2.0)
        hparams = dict(self.model.hparams)
        # Lightning omits the unpicklable default lambda when saving checkpoints.
        hparams.pop("pos_blend", None)
        self.checkpoint = {
            "state_dict": self.model.state_dict(),
            "hyper_parameters": hparams,
            "pytorch-lightning_version": lightning.__version__,
        }
        self.model.on_save_checkpoint(self.checkpoint)
        self.inferencer = object.__new__(MUNetInferencer)
        self.inferencer.plan = deepcopy(self.plan)
        self.inferencer.ckpt_path = self.root / "model.ckpt"

    def restore(self):
        torch.save(self.checkpoint, self.inferencer.ckpt_path)
        return self.inferencer._build_module().eval()

    def test_restores_all_weights_settings_and_patch_predictions(self):
        restored = self.restore()
        self.assertEqual(restored.overlap_scale, 0.0)
        self.assertEqual(restored.global_positional_encoding_proposition, 0.25)
        self.assertIs(restored.plan, self.inferencer.plan)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value)
        image = torch.randn(1, 1, 12, 8, 8)
        with torch.no_grad():
            expected = tuple(self.model(self.model.split_to_patch(image)))
            actual = tuple(restored(restored.split_to_patch(image)))
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual, strict=True):
                self.assertEqual(left[3], right[3])
                torch.testing.assert_close(left[2], right[2])
            prediction = restored.test_step({image_key: image}, 0)["out"][1]
        self.assertEqual(prediction.shape, image.shape)

    def test_rejects_plain_unet_checkpoint(self):
        self.checkpoint["state_dict"] = {
            key: value for key, value in self.checkpoint["state_dict"].items()
            if key.startswith("unet.")
        }
        with self.assertRaisesRegex(ValueError, "no MUNet bottleneck"):
            self.restore()

    def test_rejects_missing_settings(self):
        del self.checkpoint["hyper_parameters"]["overlap_scale"]
        with self.assertRaisesRegex(ValueError, "overlap_scale"):
            self.restore()

    def test_rejects_different_preprocessing_plan(self):
        self.inferencer.plan.spacing = (2, 1, 1)
        with self.assertRaisesRegex(ValueError, "spacing"):
            self.restore()

    def test_requires_all_bottleneck_weights(self):
        del self.checkpoint["state_dict"]["bottleneck.gamma"]
        with self.assertRaisesRegex(RuntimeError, "bottleneck.gamma"):
            self.restore()

    def test_explains_missing_pretrained_checkpoint(self):
        self.checkpoint["hyper_parameters"]["builder"] = (
            Builder().based_on_ckpt(self.root / "missing.ckpt").to_params()
        )
        with self.assertRaisesRegex(FileNotFoundError, "original pretrained"):
            self.restore()

    def test_all_variants_use_munet_inference(self):
        for name in ("overlap_0", "overlap_010", "bd0", "bd0g3", "bd0g4"):
            with self.subTest(name=name):
                module = importlib.import_module(
                    f"experiments.munet.experiments.{name}"
                )
                self.assertIs(module.inference, inference)


if __name__ == "__main__":
    unittest.main()
