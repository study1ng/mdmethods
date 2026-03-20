from functools import partial
from typing import Callable

from experiments.nets.plainunet import BasicBlock
from experiments.sizefreeunet.base import (
    Assertions,
    BaseUNetModule,
    Block,
    Stage,
    EncoderStage,
    DecoderStage,
    UNetHead,
    UNetStem,
    UNetEncoder,
    UNetDecoder,
    UNet,
    DownSampling,
    UpSampling,
)
from experiments.sizefreeunet.generic_modules import (
    ConvBlock,
    InstanceNormBlock,
    InterpolateUpSample,
    SequentialBlock,
    WrapperBlock,
    StridedConv,
    RepeatingBlock,
)
from torch import Tensor, nn

from experiments.utils import assert_eq, prolong


class PlainResBlock(Block):
    """Basic Block which does conv for 2 times and do residual connection"""

    def __init__(
        self,
        input_channel,
        output_channel,
        kernel_size: int | tuple[int, ...] = 3,
        *,
        dim: int,
    ):
        assert dim is not None
        self.dim = dim
        self.kernel_size = kernel_size
        super().__init__(input_channel, output_channel)
        self.conv1 = self.conv(
            input_channel, output_channel, kernel_size=self.kernel_size, dim=self.dim
        )
        self.conv2 = self.conv(
            output_channel, output_channel, kernel_size=self.kernel_size, dim=self.dim
        )
        self.resconv = self.conv(
            input_channel, output_channel, kernel_size=1, dim=self.dim
        )
        self.norm1 = self.norm(input_channel, output_channel, dim=self.dim)
        self.norm2 = self.norm(input_channel, output_channel, dim=self.dim)
        self.act1 = self.nonlin(input_channel, output_channel, dim=self.dim)
        self.act2 = self.nonlin(input_channel, output_channel, dim=self.dim)

    @property
    def norm(self):
        return InstanceNormBlock

    @property
    def nonlin(self):
        return WrapperBlock.wrap(nn.LeakyReLU(inplace=True))

    @property
    def conv(self):
        return ConvBlock

    def _forward(self, x: Tensor):
        y = self.norm2(self.conv2(self.act1(self.norm1(self.conv1(x)))))
        res = self.resconv(x)
        return self.act2(res + y)


class PlainStridedConv(DownSampling):
    """Down sampling function"""

    nonlin = staticmethod(
        lambda *args, **kwargs: nn.LeakyReLU(*args, inplace=True, **kwargs)
    )

    def __init__(
        self,
        input_channel,
        output_channel,
        pool_stride: int | tuple[int, ...] = 2,
        *,
        dim: int,
        kernel_size: int | tuple[int, ...] = 3,
    ):
        assert dim is not None
        self.dim = dim
        self.kernel_size = kernel_size
        super().__init__(input_channel, output_channel, pool_stride)
        self.conv1 = self.strided_conv(
            self.input_channel,
            self.output_channel,
            kernel_size=self.kernel_size,
            stride=self.pool_stride,
            dim=self.dim,
        )
        self.conv2 = self.conv(
            self.output_channel,
            self.output_channel,
            kernel_size=self.kernel_size,
            dim=self.dim,
        )
        self.resconv = self.strided_conv(
            self.input_channel,
            self.output_channel,
            kernel_size=1,
            stride=pool_stride,
            dim=self.dim,
        )
        self.norm1 = self.norm(self.input_channel, self.output_channel, dim=self.dim)
        self.norm2 = self.norm(self.input_channel, self.output_channel, dim=self.dim)
        self.act1 = self.nonlin(self.input_channel, self.output_channel, dim=self.dim)
        self.act2 = self.nonlin(self.input_channel, self.output_channel, dim=self.dim)

    @property
    def norm(self):
        return InstanceNormBlock

    @property
    def conv(self):
        return ConvBlock

    @property
    def strided_conv(self):
        return StridedConv

    def forward(self, x: Tensor):
        y = self.norm2(self.conv2(self.act1(self.norm1(self.conv1(x)))))
        res = self.resconv(x)
        return self.act2(res + y)


class PlainEncoderStage(EncoderStage):
    """Plain UNet Encoder Stage"""

    def __init__(
        self,
        input_channel,
        pool_channel,
        output_channel,
        kernel_size: int | tuple[int, ...] = 3,
        pool_stride: int | tuple[int, ...] = 2,
        n_blocks: int = 3,
        *,
        dim: int,
    ):
        self.kernel_size = kernel_size
        self.n_blocks = n_blocks
        self.dim = dim
        super().__init__(
            input_channel,
            pool_channel,
            output_channel,
            pool_stride=pool_stride,
        )

    def _build_block(self):
        return RepeatingBlock(
            self.after_sample_channel,
            self.output_channel,
            n_blocks=self.n_blocks,
            block_fn=PlainResBlock,
            kernel_size=self.kernel_size,
            dim=self.dim,
        )

    def _build_sample(self):
        return StridedConv(
            self.input_channel,
            self.after_sample_channel,
            kernel_size=self.kernel_size,
            pool_stride=self.pool_stride,
            dim=self.dim,
        )


class PlainDecoderStage(DecoderStage):
    """Plain UNet Decoder Stage"""

    def __init__(
        self,
        input_channel,
        pool_channel,
        output_channel,
        pool_stride: int | tuple[int, ...] = 2,
        kernel_size: int | tuple[int, ...] = 3,
        n_blocks: int = 3,
        *,
        dim: int,
    ):
        self.dim = dim
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        super().__init__(
            input_channel,
            pool_channel,
            output_channel,
            pool_stride=pool_stride,
        )

    def _build_block(self):
        return RepeatingBlock(
            self.after_sample_channel + self.skip_channel,
            self.output_channel,
            n_blocks=self.n_blocks,
            block_fn=PlainResBlock,
            kernel_size=self.kernel_size,
            dim=self.dim,
        )

    def _build_sample(self):
        return InterpolateUpSample(
            self.input_channel,
            self.after_sample_channel,
            pool_stride=self.pool_stride,
            dim=self.dim,
        )


class PlainStem(UNetStem):
    def __init__(
        self,
        input_channel,
        output_channel,
        *args,
        n_blocks: int,
        block_fn: Callable[[], Block],
        **kwargs,
    ):
        self.n_blocks = n_blocks
        self.block_fn = block_fn
        super().__init__(input_channel, output_channel)
        self.module = RepeatingBlock(
            input_channel, output_channel, *args, n_blocks=n_blocks, block_fn=block_fn, **kwargs,
        )

    def _forward(self, x):
        return self.module(x)


class PlainHead(UNetHead):
    """Plain UNet Head which is a Block"""

    def __init__(self, input_channel, output_channel, *, dim: int):
        self.output_channel = output_channel
        self.dim = dim
        super().__init__(
            input_channel=input_channel,
        )
        self.conv = ConvBlock(
            self.input_channel, self.output_channel, kernel_size=1, dim=self.dim
        )

    def _forward(self, x):
        return self.conv(x)


class PlainEncoder(UNetEncoder):
    """Plain UNet Encoder"""

    def __init__(
        self,
        n_stages,
        input_channel,
        skip_channels,
        pool_strides,
        kernel_size: tuple[tuple[int, ...], ...],
        *,
        dim: int,
    ):
        self.kernel_size = kernel_size
        self.dim = dim
        super().__init__(n_stages, input_channel, skip_channels, pool_strides)

    def _build_stem(self):
        return PlainStem(
            self.input_channel,
            self.stem_channel,
            n_blocks=3,
            block_fn=PlainResBlock,
            kernel_size=self.kernel_size[0],
            dim=self.dim,
        )

    def _build_stages(self):
        stages = []
        for i in range(self.n_stages):
            stages.append(
                PlainEncoderStage(
                    self.skip_channels[i],
                    self.skip_channels[i + 1],
                    self.skip_channels[i + 1],
                    pool_stride=self.pool_strides[i],
                    kernel_size=self.kernel_size[i + 1],
                )
            )
        return nn.ModuleList(stages)


class PlainDecoder(UNetDecoder):
    """Plain UNet Decoder"""

    def __init__(
        self,
        n_stages,
        skip_channels,
        pool_strides,
        kernel_size: tuple[tuple[int, ...], ...],
        *,
        deep_supervision=False,
        dim: int,
    ):
        self.kernel_size = kernel_size
        self.dim = dim
        super().__init__(n_stages, skip_channels, pool_strides, deep_supervision)

    def _build_head(self):
        if self.deep_supervision:
            return nn.ModuleList(
                PlainHead(skip_channel, self.output_channel, dim=self.dim)
                for skip_channel in self.skip_channels
            )
        else:
            return PlainHead(self.skip_channels[0], self.output_channel, dim=self.dim)

    def _build_stages(self):
        stages = []
        for i in range(self.n_stages):
            stages.append(
                PlainDecoderStage(
                    self.skip_channels[i + 1],
                    self.skip_channels[i],
                    self.skip_channels[i],
                    pool_stride=self.pool_strides[i],
                    kernel_size=self.conv_kernel_size[i],
                    dim=self.dim,
                )
            )
        return nn.ModuleList(stages)


class PlainUNet(UNet):
    def __init__(
        self,
        n_stages,
        input_channel,
        skip_channels,
        pool_strides=2,
        kernel_size: int | tuple[int, ...] | list[int] | list[tuple[int, ...]] = 3,
        *,
        deep_supervision=False,
        dim=None,
    ):
        self.kernel_size = kernel_size
        super().__init__(
            n_stages,
            input_channel,
            skip_channels,
            pool_strides,
            deep_supervision=deep_supervision,
            dim=dim,
        )
        if (
            isinstance(self.pool_strides, int)
            or (
                isinstance(self.pool_strides, list)
                and isinstance(self.pool_strides[0], int)
            )
        ) and self.dim is None:
            raise ValueError(
                "pool_strides is needed to be prolonged but no dim was assigned"
            )

        if isinstance(self.kernel_size, int):
            self.kernel_size: tuple = prolong(self.kernel_size, self.dim, int)

        if isinstance(self.kernel_size, tuple):
            self.kernel_size = prolong(self.kernel_size, self.n_stages - 1, list)
            self.kernel_size = [
                prolong(1, self.dim, int)
            ] + self.kernel_size  # stem doesn't need stride
        elif isinstance(self.kernel_size, list) and isinstance(
            self.kernel_size[0], int
        ):
            self.kernel_size = [prolong(ps, self.dim, int) for ps in self.kernel_size]

        assert_eq(self.n_stages + 1, len(self.kernel_size))
        self.pool_strides = tuple(self.kernel_size)

    def _build_decoder(self):
        return PlainDecoder(
            n_stages=self.n_stages,
            skip_channels=self.skip_channels,
            pool_strides=self.pool_strides,
            kernel_size=self.kernel_size,
            deep_supervision=self.deep_supervision,
            dim=self.dim,
        )

    def _build_encoder(self):
        return PlainEncoder(
            n_stages=self.n_stages,
            input_channel=self.input_channel,
            skip_channels=self.skip_channels,
            pool_strides=self.pool_strides,
            kernel_size=self.kernel_size,
            dim=self.dim,
        )
