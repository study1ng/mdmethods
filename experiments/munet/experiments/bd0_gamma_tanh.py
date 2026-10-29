"""bd0 with gamma = tanh(a), fixed c=1, and learnable a initialized to zero.

Use --experiment_name bd0_gamma_tanh with train, inference, or custom val.
For example (supply the usual bd0 paths and arguments in place of ...):
    uv run python main.py munet train --experiment_name bd0_gamma_tanh ...

All other training settings match bd0_gamma_zero. Weight decay acts on a.
The checkpoint stores a at bottleneck.parametrizations.gamma.original;
diagnostics read the effective gamma=tanh(a). Restore with this experiment
name (also when resuming), rather than the base MUNet or gamma-zero variant.
Existing MUNet checkpoints and modules are not modified.
"""

import torch
from torch.nn.utils import parametrize

from experiments.munet.experiments import bd0_gamma_zero as zero
from experiments.pretrained_seg import analyze, prune


class TanhGamma(torch.nn.Module):
    def forward(self, a):
        return torch.tanh(a)

    def right_inverse(self, gamma):
        if not torch.all(torch.isfinite(gamma) & (gamma.abs() < 1)):
            raise ValueError("Initial gamma must be finite and strictly between -1 and 1")
        return torch.atanh(gamma)


class Model(zero.Model):
    def __init__(self, builder, *, gamma=0.0, **kwargs):
        super().__init__(builder=builder, gamma=gamma, **kwargs)
        parametrize.register_parametrization(self.bottleneck, "gamma", TanhGamma())


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
