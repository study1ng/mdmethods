"""bd0 with gamma fixed at one and excluded from gradient updates.

Use --experiment_name bd0_gamma_fixed_one with train, inference, or custom val.
For example (replace ... with the usual bd0 paths and arguments):
    uv run python main.py munet train --experiment_name bd0_gamma_fixed_one ...

Start a new run from the original UNet using --pretrained_path. Other settings
match bd0_gamma_zero: frozen encoder, trainable decoder, zero overlap, and
bottleneck weight decay 0.1. AdamW skips gamma because it has no gradient.
Restore this experiment's checkpoints with the same experiment name so that
gamma remains frozen. No new dependencies or data layout changes are needed.
"""

from experiments.munet.experiments import bd0_gamma_zero as zero
from experiments.pretrained_seg import analyze, prune


class Model(zero.Model):
    def __init__(self, builder, *, gamma=1.0, **kwargs):
        # The inherited training factory passes gamma=0; this variant always
        # constructs gamma=1, including when restored from its checkpoints.
        super().__init__(builder=builder, gamma=1.0, **kwargs)
        self.gamma = 1.0
        self.hparams["gamma"] = 1.0
        self.bottleneck.gamma.requires_grad_(False)


class BottleneckSeg(zero.BottleneckSeg):
    model_class = Model


class Inferencer(zero.Inferencer):
    model_class = Model


class Custom(zero.Custom):
    model_class = Model


def train(args, parsed):
    BottleneckSeg(args, parsed)()


def inference(args, meta):
    Inferencer(args, meta)()


def custom(args, meta):
    zero.run_custom(args, meta, Custom)
