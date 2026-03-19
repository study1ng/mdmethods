from experiments.nets.baseunet import (
    Block,
    EncoderStage,
    DecoderStage,
    UNetHead,
    UNetEncoder,
    UNetDecoder,
    UNet,
)
from torch import nn, Tensor
import torch.nn.functional as F
from experiments.nets.generic_blocks import SequentialBlock


def pad(kernel_size: int | tuple[int, ...]) -> int | tuple[int, ...]:
    if isinstance(kernel_size, int):
        assert (kernel_size & 1) == 1, "kernel size should be odd"
        return kernel_size // 2
    else:
        rem = ((ks & 1) == 1 for ks in kernel_size)
        assert all(rem), "kernel size should be odd"
        return tuple(ks // 2 for ks in kernel_size)


def conv(dim: int):
    match dim:
        case 1:
            return nn.Conv1d
        case 2:
            return nn.Conv2d
        case 3:
            return nn.Conv3d
        case _:
            raise ValueError(f"dim {dim} should be 1~3")


def instance_norm(dim: int):
    match dim:
        case 1:
            return nn.InstanceNorm1d
        case 2:
            return nn.InstanceNorm2d
        case 3:
            return nn.InstanceNorm3d
        case _:
            raise ValueError(f"dim {dim} should be 1~3")


class SingleConvBlock(Block):
    def __init__(
        self,
        input_channel,
        output_channel,
        size,
        kernel_size: int | tuple[int, ...] = 3,
        *args,
        **kwargs,
    ):
        self.kernel_size = kernel_size
        self.padding = pad(kernel_size)
        super().__init__(input_channel, output_channel, size)
        self.conv = conv(self.dim)(
            input_channel,
            output_channel,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            *args,
            **kwargs,
        )

    def _forward(self, x):
        return self.conv(x)


class BasicBlock(Block):
    nonlin = staticmethod(
        lambda *args, **kwargs: nn.LeakyReLU(*args, inplace=True, **kwargs)
    )
    conv = SingleConvBlock

    def __init__(
        self,
        input_channel,
        output_channel,
        size,
        kernel_size: int | tuple[int, ...] = 3,
    ):
        self.kernel_size = kernel_size
        super().__init__(input_channel, output_channel, size)
        self.conv1 = self.conv(
            input_channel, output_channel, size, kernel_size=kernel_size
        )
        self.conv2 = self.conv(
            output_channel, output_channel, size, kernel_size=kernel_size
        )
        self.resconv = self.conv(input_channel, output_channel, size, kernel_size=1)
        self.norm1 = self.norm(output_channel)
        self.norm2 = self.norm(output_channel)
        self.act1 = self.nonlin()
        self.act2 = self.nonlin()

    @property
    def norm(self):
        return instance_norm(self.dim)

    def _forward(self, x: Tensor):
        y = self.norm2(self.conv2(self.act1(self.norm1(self.conv1(x)))))
        res = self.resconv(x)
        return self.act2(res + y)


class BasicStridedConv(nn.Module):
    nonlin = staticmethod(
        lambda *args, **kwargs: nn.LeakyReLU(*args, inplace=True, **kwargs)
    )

    def __init__(
        self,
        input_channel,
        output_channel,
        dim: int,
        kernel_size: int | tuple[int, ...] = 3,
        pool_stride: int | tuple[int, ...] = 2,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = pad(kernel_size)
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.dim = dim
        self.pool_stride = pool_stride

        self.conv1 = self.conv(
            input_channel,
            output_channel,
            kernel_size=kernel_size,
            padding=self.padding,
            stride=pool_stride,
        )
        self.conv2 = self.conv(
            output_channel,
            output_channel,
            kernel_size=kernel_size,
            padding=self.padding,
        )
        self.resconv = self.conv(
            input_channel, output_channel, kernel_size=1, stride=pool_stride,
        )
        self.norm1 = self.norm(output_channel)
        self.norm2 = self.norm(output_channel)
        self.act1 = self.nonlin()
        self.act2 = self.nonlin()

    @property
    def norm(self):
        return instance_norm(self.dim)

    @property
    def conv(self):
        return conv(self.dim)

    def forward(self, x: Tensor):
        y = self.norm2(self.conv2(self.act1(self.norm1(self.conv1(x)))))
        res = self.resconv(x)
        return self.act2(res + y)


class UpsampleLayer(nn.Module):
    def __init__(
        self,
        input_channel,
        output_channel,
        dim: int = 3,
        pool_stride: int | tuple[int, ...] = 2,
        mode="nearest",
    ):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.dim = dim
        self.pool_stride = pool_stride
        self.mode = mode
        self.conv_layer = self.conv(input_channel, output_channel, kernel_size=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_stride, mode=self.mode)
        x = self.conv_layer(x)
        return x

    @property
    def conv(self):
        return conv(self.dim)


class RepeatingBlock(Block):
    def __init__(
        self, input_channel, output_channel, size, n_blocks: int, kernel_size: int | tuple[int, ...] = 3, block_fn=BasicBlock
    ):
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        super().__init__(input_channel, output_channel, size)
        self.module = SequentialBlock(
            block_fn(input_channel, output_channel, size, kernel_size=kernel_size),
            *(
                block_fn(output_channel, output_channel, size, kernel_size=kernel_size)
                for _ in range(n_blocks - 1)
            ),
        )

    def _forward(self, x):
        return self.module(x)


class PlainEncoderStage(EncoderStage):
    def __init__(
        self,
        input_channel,
        after_sample_channel,
        output_channel,
        input_size,
        output_size,
        pool_stride: int | tuple[int, ...] = 2,
        kernel_size: int | tuple[int, ...] = 3,
        n_blocks: int = 3,
    ):
        self.kernel_size = kernel_size
        self.n_blocks = n_blocks
        super().__init__(
            input_channel,
            after_sample_channel,
            output_channel,
            input_size,
            output_size,
            pool_stride=pool_stride,
        )

    def _build_block(self):
        return RepeatingBlock(
            self.after_sample_channel,
            self.output_channel,
            self.output_size,
            kernel_size = self.kernel_size,
            n_blocks=self.n_blocks,
        )

    def _build_sample(self):
        return BasicStridedConv(
            self.input_channel,
            self.after_sample_channel,
            dim=self.dim,
            pool_stride=self.pool_stride,
            kernel_size=self.kernel_size
        )


class PlainDecoderStage(DecoderStage):
    def __init__(
        self,
        input_channel,
        after_sample_channel,
        skip_channel,
        output_channel,
        input_size,
        output_size,
        pool_stride: int | tuple[int, ...] = 2,
        kernel_size: int | tuple[int, ...] = 3,
        n_blocks: int = 3,
    ):
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        super().__init__(
            input_channel,
            after_sample_channel,
            skip_channel,
            output_channel,
            input_size,
            output_size,
            pool_stride=pool_stride,
        )

    def _build_block(self):
        return RepeatingBlock(
            self.after_sample_channel + self.skip_channel,
            self.output_channel,
            self.output_size,
            n_blocks=self.n_blocks,
            kernel_size=self.kernel_size
        )

    def _build_sample(self):
        return UpsampleLayer(
            self.input_channel,
            self.after_sample_channel,
            dim=self.dim,
            pool_stride=self.pool_stride,
        )


BasicStem = RepeatingBlock


class BasicHead(UNetHead):
    def __init__(self, input_channel, output_channel, size):
        super().__init__(input_size=size, input_channel=input_channel, output_size=size, output_channel=output_channel)
        self.conv = conv(self.dim)(input_channel, output_channel, kernel_size=1)

    def _forward(self, x):
        return self.conv(x)


class PlainEncoder(UNetEncoder):
    def _build_stem(self):
        return BasicStem(
            self.input_channel, self.stem_channel, self.input_size, n_blocks=3,
        )

    def _build_stages(self):
        stages = []
        for i in range(self.n_stages):
            stages.append(
                PlainEncoderStage(
                    self.skip_channels[i],
                    self.skip_channels[i] * self.pool_channel_increase_ratio[i],
                    self.skip_channels[i + 1],
                    self.skip_size[i],
                    self.skip_size[i + 1],
                    pool_stride=self.pool_strides[i],
                    kernel_size = self.conv_kernel_size[i]
                )
            )
        return nn.ModuleList(stages)


class PlainDecoder(UNetDecoder):
    def _build_head(self):
        if self.deep_supervision:
            return nn.ModuleList(
                BasicHead(skip_channel, self.output_channel, size=skip_size)
                for skip_channel, skip_size in zip(self.skip_channels, self.skip_size)
            )
        else:
            return BasicHead(
                self.skip_channels[0], self.output_channel, size=self.skip_size[0]
            )

    def _build_stages(self):
        stages = []
        for i in range(self.n_stages):
            stages.append(
                PlainDecoderStage(
                    self.skip_channels[i + 1],
                    self.skip_channels[i],
                    self.skip_channels[i],
                    self.skip_channels[i],
                    self.skip_size[i + 1],
                    self.skip_size[i],
                    pool_stride=self.pool_strides[i],
                    kernel_size=self.conv_kernel_size[i]
                )
            )
        return nn.ModuleList(stages)


class PlainUNet(UNet):
    def _build_decoder(self):
        return PlainDecoder(
            self.output_channel,
            self.skip_size,
            self.skip_channels,
            self.n_stages,
            self.conv_kernel_size,
            self.pool_strides,
            self.pool_channel_increase_ratio,
            self.deep_supervision,
        )

    def _build_encoder(self):
        return PlainEncoder(
            self.patch_size,
            self.patch_channel,
            self.stem_channel,
            self.n_stages,
            self.skip_channels,
            self.skip_size,
            self.conv_kernel_size,
            self.pool_strides,
            self.pool_channel_increase_ratio,
        )
