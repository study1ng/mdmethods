from fractions import Fraction
from sys import maxsize

from experiments.nets.base import Block
import torch
from torch import nn, autocast
from mamba_ssm import Mamba
from experiments.nets.plainunet import (
    PlainEncoderStage,
    PlainEncoder,
    PlainUNet,
)
from experiments.nets.generic_modules import SequentialBlock


class BiMambaBlock(Block):
    """A block which use Bidirectional Mamba"""

    def __init__(
        self,
        input_channel,
        output_channel,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__(input_channel, output_channel)
        print(f"MambaLayer: dim: {input_channel}")
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.norm = nn.LayerNorm(input_channel)
        self.mambas = nn.ModuleList(
            [Mamba(input_channel, d_state, d_conv, expand) for _ in range(2)]
        )
        self.out_proj = nn.Linear(input_channel * 2, output_channel)

    def forward_patch_token(self, x):
        B, d_model = x.shape[:2]
        assert d_model == self.input_channel
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        # Bi-directional scan
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mambas[0](x_norm)
        x_flip = x_norm.flip([1]).contiguous()
        x_flip_mamba = self.mambas[1](x_flip).flip([1]).contiguous()
        x_mamba = torch.cat([x_mamba, x_flip_mamba], dim=-1)
        x_mamba = self.out_proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_channel, *img_dims)
        return out

    @autocast(device_type="cuda", enabled=False)
    def _forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)

        out = self.forward_patch_token(x)

        return out


class MEncoderStage(PlainEncoderStage):
    """Encoder Stage for BiMamba"""

    def __init__(
        self,
        input_channel,
        pool_channel,
        output_channel,
        kernel_size,
        pool_scale,
        n_blocks=3,
        d_state=16,
        d_conv=4,
        expand=2,
        *,
        dim: int,
    ):
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dim = dim
        assert n_blocks > 1, f"n_blocks should be greater than 1"
        super().__init__(
            input_channel,
            pool_channel,
            output_channel,
            kernel_size,
            pool_scale,
            n_blocks=n_blocks - 1,
            dim=self.dim,
        )

    def _build_block(self):
        return SequentialBlock(
            super()._build_block(),
            BiMambaBlock(
                self.output_channel,
                self.output_channel,
                self.d_state,
                self.d_conv,
                self.expand,
            ),
        )


class MEncoder(PlainEncoder):
    """Encoder for BiMamba"""

    def __init__(
        self,
        n_stages,
        input_channel,
        skip_channels,
        pool_scales,
        kernel_size: tuple[tuple[int, ...], ...],
        *,
        dim: int,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        super().__init__(
            n_stages=n_stages,
            input_channel=input_channel,
            skip_channels=skip_channels,
            pool_scales=pool_scales,
            kernel_size=kernel_size,
            dim=dim,
        )

    def _build_stages(self):
        stages = []
        for i in range(self.n_stages):
            stages.append(
                MEncoderStage(
                    self.skip_channels[i],
                    self.skip_channels[i + 1],
                    self.skip_channels[i + 1],
                    pool_scale=self.pool_scales[i],
                    kernel_size=self.kernel_size[i + 1],
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand=self.expand,
                    dim=self.dim,
                )
            )
        return nn.ModuleList(stages)


class UBiMamba(PlainUNet):
    def __init__(
        self,
        n_stages,
        input_channel,
        skip_channels,
        output_channel,
        decoder_pool_scales=Fraction(2),
        kernel_size=3,
        *,
        deep_supervision=False,
        dim,
        feature_channel_limitation=maxsize,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        super().__init__(
            n_stages,
            input_channel,
            skip_channels,
            output_channel,
            decoder_pool_scales,
            kernel_size,
            deep_supervision=deep_supervision,
            dim=dim,
            feature_channel_limitation=feature_channel_limitation,
        )

    def _build_encoder(self):
        return MEncoder(
            n_stages=self.n_stages,
            input_channel=self.input_channel,
            skip_channels=self.skip_channels,
            pool_scales=self.encoder_pool_scales,
            kernel_size=self.kernel_size,
            dim=self.dim,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )
