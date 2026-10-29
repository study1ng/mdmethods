"""bd0 with an unconstrained scalar gamma initialized to zero.

Use --experiment_name bd0_gamma_zero with train, inference, or custom val.
For example (supply the usual bd0 paths and arguments in place of ...):
    uv run python main.py munet train --experiment_name bd0_gamma_zero ...

Start a new run with --pretrained_path pointing to the original UNet.
As in bd0, the encoder is frozen, the decoder is trainable, overlap is zero,
and bottleneck weight decay is 0.1 (including gamma). No data layout changes.
"""

import argparse

import torch

from experiments.munet import MUNetCustom, MUNetInferencer, MUNetMetric
from experiments.munet.experiments.bd0 import BottleneckSeg as BaseBottleneckSeg
from experiments.munet.model import MUNetTrainingModule
from experiments.nets.builder import Builder
from experiments.plan import Plan
from experiments.pretrained_seg import analyze, prune


class Model(MUNetTrainingModule):
    def __init__(self, builder, *, gamma=0.0, **kwargs):
        super().__init__(builder=builder, gamma=gamma, **kwargs)


class BottleneckSeg(BaseBottleneckSeg):
    model_class = Model

    def _build_module(self):
        if self.args.pretrained_path is None:
            raise ValueError("Munet needs pretrained model")
        builder = Builder().based_on_ckpt(self.args.pretrained_path).to_params()
        return self.model_class(
            builder=builder, plan=self.plan, overlap_scale=0.0, gamma=0.0,
        )


class ExperimentRestoreMixin:
    """Restore the experiment model while retaining MUNet inference options."""

    model_class = Model

    def _build_module(self):
        checkpoint = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("Expected a MUNet Lightning checkpoint.")
        state = checkpoint.get("state_dict", {})
        if not any(key.startswith("bottleneck.") for key in state):
            raise ValueError("The checkpoint has no MUNet bottleneck weights.")
        hparams = checkpoint.get("hyper_parameters", {})
        required = (
            "builder", "plan", "overlap_scale",
            "global_positional_encoding_proposition", "gamma",
        )
        missing = [key for key in required if key not in hparams]
        if missing:
            raise ValueError(f"MUNet checkpoint is missing hyperparameters: {missing}")
        saved_plan = hparams["plan"]
        if not isinstance(saved_plan, Plan):
            raise ValueError("The checkpoint must contain the training Plan object.")
        plan_fields = (
            "patch_size", "spacing", "mean", "std", "percentile_00_5",
            "percentile_99_5", "pool_strides", "conv_kernel_size", "stem_channel",
            "max_feature_channel", "n_stages", "dim",
        )
        mismatched = [
            field for field in plan_fields
            if not hasattr(saved_plan, field)
            or getattr(saved_plan, field) != getattr(self.plan, field)
        ]
        if mismatched:
            raise ValueError(
                f"Inference plan differs from the training plan: {mismatched}. "
                "Use the plan used to train this checkpoint."
            )
        del state, hparams, saved_plan, checkpoint
        module = self.model_class.load_from_checkpoint(
            self.ckpt_path, map_location="cpu", strict=True,
            weights_only=False, plan=self.plan,
        )
        module.no_mixing = self.args.no_mixing
        module.no_ppe = self.args.no_ppe
        return module


class Inferencer(ExperimentRestoreMixin, MUNetInferencer):
    pass


class Custom(ExperimentRestoreMixin, MUNetCustom):
    pass


def train(args, parsed):
    BottleneckSeg(args, parsed)()


def inference(args, meta):
    Inferencer(args, meta)()


def run_custom(args, meta, custom_class):
    if args and args[0] == "observe":
        from experiments.munet.observe import observe
        return observe(args[1:], meta, custom_class)
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["val", "metric", "observe"])
    parser.add_argument("-mh", "--module-help", action="help")
    if args and args[0] in {"val", "metric"}:
        action, remaining = args[0], args[1:]
    else:
        parsed, remaining = parser.parse_known_args(args)
        action = parsed.action
    if action == "metric":
        MUNetMetric(remaining, meta)()
    else:
        custom_class(["val", *remaining], meta)()


def custom(args, meta):
    run_custom(args, meta, Custom)
