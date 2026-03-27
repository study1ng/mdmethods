from fractions import Fraction
from typing import Callable

from experiments.assertions import AssertEq
from experiments.nets.base import BaseUNetModule, Block, Pool
from torch import nn, Tensor

from experiments.utils import assert_to_integer, element_wise, reciprocal, repeat
import torch.nn.functional as F


class SequentialBlock(Block):
    """nn.Sequential for block version

    Parameters
    ----------
    Block : Block
        blocks
    """

    def __init__(self, *blocks: Block):
        assert len(blocks) != 0, f"Sequential Block needs at least 1 block"
        super().__init__(blocks[0].input_channel, blocks[-1].output_channel)
        self.blocks = nn.Sequential(*blocks)

    def _forward(self, x: Tensor):
        y = self.blocks(x)
        return y


class InstanceNormBlock(Block):
    def __init__(self, input_channel, *, dim: int):
        self.dim = dim
        super().__init__(input_channel, input_channel)
        self.module = self.instance_norm(self.dim)(self.input_channel)

    def _forward(self, x):
        return self.module(x)

    @staticmethod
    def instance_norm(dim: int):
        """get the corresponding InstanceNorm by dim

        Parameters
        ----------
        dim : int
            the dimension. select from 1~3

        Returns
        -------
        nn.Module

        Raises
        ------
        ValueError
            if dim not in {1,2,3}
        """
        match dim:
            case 1:
                return nn.InstanceNorm1d
            case 2:
                return nn.InstanceNorm2d
            case 3:
                return nn.InstanceNorm3d
            case _:
                raise ValueError(f"dim {dim} should be 1~3")


class WrapperBlock(Block):
    def __init__(self, input_channel, output_channel, module: nn.Module):
        super().__init__(input_channel, output_channel)
        self.module = module

    def _forward(self, x):
        return self.module(x)

    @classmethod
    def wrap(cls, module: nn.Module):
        return lambda input_channel, output_channel: cls(
            input_channel, output_channel, module
        )


class ConvBlock(Block):
    def __init__(
        self,
        input_channel,
        output_channel,
        kernel_size: int | tuple[int, ...] = 3,
        *,
        dim: int | None = None,
    ):
        assert dim is not None or isinstance(kernel_size, tuple), "dim is ambiguous"
        self.dim = dim if dim is not None else len(kernel_size)
        self.kernel_size = kernel_size

        @element_wise(int)
        def _assert_odd(i):
            if i % 2 == 0:
                raise ValueError(f"Expected odd kernel size got {self.kernel_size}")

        _assert_odd(self.kernel_size)
        super().__init__(input_channel, output_channel)
        self.module = self.conv(self.dim)(
            self.input_channel, self.output_channel, self.kernel_size, padding="same"
        )

    def _forward(self, x):
        return self.module(x)

    @staticmethod
    def conv(dim: int):
        """get the corresponding Conv by dim

        Parameters
        ----------
        dim : int
            the dimension. select from 1~3

        Returns
        -------
        nn.Module

        Raises
        ------
        ValueError
            if dim not in {1,2,3}
        """
        match dim:
            case 1:
                return nn.Conv1d
            case 2:
                return nn.Conv2d
            case 3:
                return nn.Conv3d
            case _:
                raise ValueError(f"dim {dim} should be 1~3")


class StridedConv(Pool):
    def __init__(
        self,
        input_channel,
        output_channel,
        kernel_size: int | tuple[int, ...] = 3,
        pool_scale: Fraction | tuple[Fraction, ...] = Fraction(2),
        *,
        dim: int,
    ):
        assert dim is not None or isinstance(kernel_size, tuple), "dim is ambiguous"
        self.dim = dim if dim is not None else len(kernel_size)
        self.kernel_size = kernel_size
        super().__init__(input_channel, output_channel, pool_scale)

        try:
            self.pool_stride = assert_to_integer(reciprocal(pool_scale))
        except AssertionError:
            raise AssertionError(f"Strided Conv only accepts reciprocal of integer")
        
        self.module = ConvBlock.conv(self.dim)(
            self.input_channel,
            self.output_channel,
            self.kernel_size,
            stride=self.pool_stride,
            padding=self.pad(self.kernel_size),
        )

    def _forward(self, x):
        return self.module(x)

    @staticmethod
    @element_wise(int)
    def pad(kernel_size: int) -> int:
        """calculate the pad size from conv kernel size

        Parameters
        ----------
        kernel_size : int
            kernel size

        Returns
        -------
        int
            padding size
        """
        assert (kernel_size & 1) == 1, "kernel size should be odd"
        return kernel_size // 2


class InterpolateUpSample(Pool):
    """up sampling function"""

    def __init__(
        self,
        input_channel,
        output_channel,
        pool_scale: Fraction | tuple[Fraction, ...] = Fraction(2),
        *,
        dim: int = 3,
        mode="nearest",
    ):
        self.dim = dim
        self.mode = mode
        super().__init__(input_channel, output_channel, pool_scale)
        try:
            self.pool_stride = assert_to_integer(pool_scale)
        except AssertionError:
            raise AssertionError("Interpolate Up Sample only accepts integer pool scale")
        self.module = self.conv(input_channel, output_channel, 1, dim=self.dim)

    def _forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_stride, mode=self.mode)
        x = self.module(x)
        return x

    @property
    def conv(self):
        return ConvBlock


class RepeatingBlock(Block):
    """repeat a block

    Parameters
    ----------
    n_blocks : int
        how many times to repeat
    block_fn : Function
        function which get no argument and return a Block
    """

    def __init__(
        self,
        input_channel,
        output_channel,
        *args,
        n_blocks: int,
        block_fn: Callable[..., Block],
        **kwargs,
    ):
        self.n_blocks = n_blocks
        self.block_fn = block_fn
        assert self.n_blocks > 0, f"n_blocks {self.n_blocks} <= 0"
        super().__init__(input_channel, output_channel)
        self.module = SequentialBlock(
            block_fn(input_channel, output_channel, *args, **kwargs),
            *(
                block_fn(output_channel, output_channel, *args, **kwargs)
                for _ in range(n_blocks - 1)
            ),
        )

    def _forward(self, x):
        return self.module(x)

class GlobalAverageGap(BaseUNetModule):
    def __init__(self, output_size: int | tuple[int, ...], *, dim: int):
        super().__init__()
        self.dim = dim
        self.output_size = repeat(output_size, dim, types=int)
        AssertEq()(self.dim, len(self.output_size))
        self.module = self.gap(self.dim)(self.output_shape)


    @staticmethod
    def gap(dim: int):
        match dim:
            case 1:
                return nn.AdaptiveAvgPool1d
            case 2:
                return nn.AdaptiveAvgPool2d
            case 3:
                return nn.AdaptiveAvgPool3d            
            case _:
                raise ValueError(f"dim {dim} should be 1~3")


    def _forward(self, x):
        return self.module(x)