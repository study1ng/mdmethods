import lightning as L
import torch, math, einops.layers.torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from functools import partial
from experiments.nets import UNet, UNetDecoder, UNetHead, Block, Stage
from experiments.config import image_key
from experiments.plan import Plan
from experiments.nets.generic_blocks import SequentialBlock
from experiments.nets.plainunet import SingleConvBlock
from math import prod
from experiments.utils import assert_divisable, assert_eq

def pixelshuffle(dim: int, *scales: int) -> einops.layers.torch.Rearrange:
    assert_eq(dim, len(scales))
    scale_characters = [f"p{i}" for i in range(dim)]
    scale_pattern = " ".join(scale_characters)
    dimensions = [f"d{i}" for i in range(dim)]
    dimensions_pattern = " ".join(dimensions)
    scale_dims = [f"{(scale_characters[i])}, {dimensions[i]})" for i in range(dim)]
    scale_dims_pattern = " ".join(scale_dims)

    scale_dict = {scale_characters[i]: scales[i] for i in range(dim)}

    pattern = f"b ({scale_pattern} c) {dimensions_pattern} -> b c {scale_dims_pattern}"
    return einops.layers.torch.Rearrange(pattern, **scale_dict)


def generate_mask(
    image: torch.Tensor,
    mask_shape: tuple[int, int, int],
    mask_ratio: float,
    device: str | None = None,
) -> torch.Tensor:
    sh = image.shape
    assert len(sh) == 5
    assert sh[2] % mask_shape[0] == 0
    assert sh[3] % mask_shape[1] == 0
    assert sh[4] % mask_shape[2] == 0
    msh = (sh[0], 1, *[sh[i + 2] // mask_shape[i] for i in range(3)])
    mask = torch.rand(msh, device=device) < mask_ratio
    mask = F.interpolate(mask.float(), sh[2:], mode="nearest")
    return mask


class PixelShuffleHead(UNetHead):
    """PixelShuffleHeadは入力特徴マップに対してピクセルシャッフルを行うことによって期待する解像度, チャネル数に戻すヘッドである"""

    def __init__(self, input_size, input_channel, output_size, output_channel):
        super().__init__(input_size, input_channel, output_size, output_channel)
        divided = prod(assert_divisable(output_size, input_size))
        self.ps = nn.Sequential(
            SequentialBlock(
                SingleConvBlock(
                    input_channel, input_channel, input_size, kernel_size=3
                ),
                SingleConvBlock(
                    input_channel, output_channel, input_size, kernel_size=1
                ),  # 必要なチャネル数はmath.prod(output_channel_size // input_channel_size)
            ),
            pixelshuffle(self.dim, *divided)
        )

    def _forward(self, x):
        return self.ps(x)
