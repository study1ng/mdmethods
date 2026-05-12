from experiments.nets.generic_modules import ConvBlock
from experiments.nets.plainunet import PlainHead
from experiments.nets.ubimamba import BiMambaBlock
from torch import nn

class MambaConvHead(PlainHead):
    def __init__(self, input_channel, output_channel, *, dim):
        super().__init__(
            input_channel=input_channel,
            output_channel=output_channel,
            dim=dim
        )
        self.mamba = BiMambaBlock(self.input_channel, self.input_channel)

    def _forward(self, x):
        return self.conv(self.mamba(x))

class C2Head(PlainHead):
    def __init__(self, input_channel, output_channel, *, dim):
        super().__init__(
            input_channel=input_channel,
            output_channel=output_channel,
            dim=dim
        )
        self.c = ConvBlock(self.input_channel, self.input_channel, dim=self.dim)
        self.act = nn.LeakyReLU()

    def _forward(self, x):
        return self.conv(self.act(self.c(x)))
