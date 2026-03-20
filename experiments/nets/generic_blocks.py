from experiments.nets.baseunet import Block
from torch import nn, Tensor


class SequentialBlock(Block):
    """nn.Sequential for block version

    Parameters
    ----------
    Block : Block
        blocks
    """
    def __init__(self, *blocks: Block):
        assert len(blocks) != 0, f"Sequential Block needs more than 1 block"
        super().__init__(blocks[0].input_channel, blocks[-1].output_channel, blocks[0].size)
        self.blocks = nn.Sequential(*blocks)

    def _forward(self, x: Tensor):
        y = self.blocks(x)
        return y
