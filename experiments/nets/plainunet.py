from sys import maxsize
from typing import Callable

from experiments.nets.base import (
    Block,
    EncoderStage,
    DecoderStage,
    UNetHead,
    UNetStem,
    UNetEncoder,
    UNetDecoder,
    UNet,
    DownSampling,
)
from experiments.nets.generic_modules import (
    ConvBlock,
    InstanceNormBlock,
    InterpolateUpSample,
    WrapperBlock,
    StridedConv,
    RepeatingBlock,
)
from torch import Tensor, nn

from experiments.utils import assert_eq, elementwise_min, repeat


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
            self.input_channel,
            self.output_channel,
            kernel_size=self.kernel_size,
            dim=self.dim,
        )
        self.conv2 = self.conv(
            self.output_channel,
            self.output_channel,
            kernel_size=self.kernel_size,
            dim=self.dim,
        )
        self.resconv = self.conv(
            self.input_channel, self.output_channel, kernel_size=1, dim=self.dim
        )
        self.norm1 = self.norm(self.output_channel, dim=self.dim)
        self.norm2 = self.norm(self.output_channel, dim=self.dim)
        self.act1 = self.nonlin(self.output_channel, self.output_channel)
        self.act2 = self.nonlin(self.output_channel, self.output_channel)

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
            pool_stride=self.pool_stride,
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
            pool_stride=self.pool_stride,
            dim=self.dim,
        )
        self.norm1 = self.norm(self.output_channel, dim=self.dim)
        self.norm2 = self.norm(self.output_channel, dim=self.dim)
        self.act1 = self.nonlin(self.output_channel, self.output_channel)
        self.act2 = self.nonlin(self.output_channel, self.output_channel)

    @property
    def norm(self):
        return InstanceNormBlock

    @property
    def conv(self):
        return ConvBlock

    @property
    def strided_conv(self):
        return StridedConv

    @property
    def nonlin(self):
        return WrapperBlock.wrap(nn.LeakyReLU(inplace=True))

    def _forward(self, x: Tensor):
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
            self.pool_channel,
            self.output_channel,
            n_blocks=self.n_blocks,
            block_fn=PlainResBlock,
            kernel_size=self.kernel_size,
            dim=self.dim,
        )

    def _build_pool(self):
        return StridedConv(
            self.input_channel,
            self.pool_channel,
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
        skip_channel,
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
            skip_channel,
            output_channel,
            pool_stride=pool_stride,
        )

    def _build_block(self):
        return RepeatingBlock(
            self.pool_channel + self.skip_channel,
            self.output_channel,
            n_blocks=self.n_blocks,
            block_fn=PlainResBlock,
            kernel_size=self.kernel_size,
            dim=self.dim,
        )

    def _build_pool(self):
        return InterpolateUpSample(
            self.input_channel,
            self.pool_channel,
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
            input_channel,
            output_channel,
            *args,
            n_blocks=n_blocks,
            block_fn=block_fn,
            **kwargs,
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

    @classmethod
    def attach_to_unet(cls, unet: "PlainUNet"):
        if unet.deep_supervision:
            unet.decoder.head = nn.ModuleList(
                PlainHead(skip_channel, unet.output_channel, dim=unet.dim)
                for skip_channel in unet.skip_channels
            )
        else:
            unet.decoder.head = PlainHead(
                unet.skip_channels[0], unet.output_channel, dim=unet.dim
            )


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
                    dim=self.dim,
                )
            )
        return nn.ModuleList(stages)


class PlainDecoder(UNetDecoder):
    """Plain UNet Decoder"""

    def __init__(
        self,
        n_stages,
        skip_channels,
        output_channel,
        pool_strides,
        kernel_size: tuple[tuple[int, ...], ...],
        *,
        deep_supervision=False,
        dim: int,
    ):
        self.kernel_size = kernel_size
        self.dim = dim
        self.output_channel = output_channel
        super().__init__(
            n_stages, skip_channels, pool_strides, deep_supervision=deep_supervision
        )

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
                    self.skip_channels[i],
                    pool_stride=self.pool_strides[i],
                    kernel_size=self.kernel_size[i + 1],
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
        output_channel: int,
        pool_strides=2,
        kernel_size: int | tuple[int, ...] | list[int] | list[tuple[int, ...]] = 3,
        *,
        deep_supervision=False,
        dim: int,
        feature_channel_limitation: int = maxsize,
    ):
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.feature_channel_limitation = feature_channel_limitation

        if isinstance(skip_channels, (int, tuple)):
            skip_channels = elementwise_min(skip_channels)
        else:
            skip_channels = [elementwise_min(sc) for sc in skip_channels]

        if isinstance(self.kernel_size, int):
            self.kernel_size = repeat(self.kernel_size, self.dim)
        if isinstance(self.kernel_size, tuple):
            self.kernel_size = repeat(
                self.kernel_size, self.n_stages + 1, wrap_type=list
            )
        elif isinstance(self.kernel_size, list) and isinstance(
            self.kernel_size[0], int
        ):
            self.kernel_size = [repeat(ps, self.dim) for ps in self.kernel_size]

        assert_eq(self.n_stages + 1, len(self.kernel_size))
        self.kernel_size = tuple(self.kernel_size)
        super().__init__(
            n_stages,
            input_channel,
            skip_channels,
            pool_strides,
            deep_supervision=deep_supervision,
            dim=dim,
        )

    def _build_decoder(self):
        return PlainDecoder(
            n_stages=self.n_stages,
            skip_channels=self.skip_channels,
            output_channel=self.output_channel,
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

    @staticmethod
    def _plan_to_arguments(
        plan,
        input_channel,
        output_channel,
        skip_channel_ratio=2,
        deep_supervision=False,
        **kwargs,
    ):
        args = UNet._plan_to_arguments(
            plan, input_channel, skip_channel_ratio, deep_supervision, **kwargs
        )
        args["output_channel"] = output_channel
        args["feature_channel_limitation"] = plan.max_feature_channel
        args["kernel_size"] = plan.conv_kernel_size
        args.update(kwargs)
        return args
