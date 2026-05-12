from torch import nn
import torch.nn.functional as F
import torch

from experiments.nets.base import UNetHead, Block
from experiments.nets.generic_modules import (
    ConvBlock,
    GlobalAveragePool,
    LinearBlock,
    RepeatingBlock,
    SequentialBlock,
)


class DinoHeadBlock(Block):
    def __init__(self, input_channel, output_channel, *, dim):
        self.lin = ConvBlock(input_channel, output_channel, kernel_size=1, dim=dim)
        self.nonlin = nn.GELU()

    def _forward(self, x):
        return self.nonlin(self.lin(x))



class DinoHead(UNetHead):
    """DinoHead
    To Study Dense Feature, We use Conv1d instead of Linear
    """

    def __init__(
        self,
        input_channel,
        output_channel,
        hidden_channel=2048,
        bottleneck_channel=256,
        *,
        dim
    ):
        super().__init__(input_channel)
        self.output_channel = output_channel
        self.hidden_channel = hidden_channel
        self.bottleneck_channel = bottleneck_channel
        self.dim = dim

        self.gap = GlobalAveragePool(output_size=1, dim=self.dim)
        self.l = nn.Sequential(
            DinoHeadBlock(self.input_channel, self.hidden_channel),
            DinoHeadBlock(self.hidden_channel, self.hidden_channel),
            ConvBlock(self.hidden_channel, self.bottleneck_channel),
        )
        self.output = nn.utils.parametrizations.weight_norm(
            LinearBlock(self.bottleneck_channel, self.output_channel, bias=False)
        )

    def _forward(self, x):
        g = torch.flatten(self.gap(x), start_dim=1)
        l = self.l(g)
        n = F.normalize(l, dim=1)
        o = self.output(n)
        return o
