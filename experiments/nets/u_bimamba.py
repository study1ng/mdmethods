from torch import autocast, nn
import torch
from mamba_ssm import Mamba
from experiments.nets.baseunet import Block
from experiments.nets.plainunet import PlainEncoderStage, PlainEncoder, PlainUNet
from math import prod


class MambaBlock(Block):
    def __init__(
        self,
        input_channel,
        output_channel,
        size,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__(input_channel, output_channel, size)
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
    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)

        out = self.forward_patch_token(x)

        return out


class MEncoderStage(PlainEncoderStage):
    def __init__(
        self,
        input_channel,
        after_sample_channel,
        output_channel,
        input_size,
        output_size,
        pool_stride=2,
        kernel_size=3,
        n_blocks=3,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        super().__init__(
            input_channel,
            after_sample_channel,
            output_channel,
            input_size,
            output_size,
            pool_stride,
            kernel_size,
            n_blocks,
        )

    def _build_block(self):
        return nn.Sequential(
            super()._build_block(),
            MambaBlock(
                self.output_channel,
                self.output_channel,
                self.output_size,
                self.d_state,
                self.d_conv,
                self.expand,
            ),
        )


class MEncoder(PlainEncoder):
    def __init__(
        self,
        input_size,
        input_channel,
        stem_channel,
        n_stages,
        skip_channels,
        skip_size,
        conv_kernel_size=3,
        pool_strides=2,
        pool_channel_increase_ratio=2,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        super().__init__(
            input_size,
            input_channel,
            stem_channel,
            n_stages,
            skip_channels,
            skip_size,
            conv_kernel_size,
            pool_strides,
            pool_channel_increase_ratio,
        )

    def _build_stages(self):
        stages = []
        for i in range(self.n_stages):
            stages.append(
                MEncoderStage(
                    self.skip_channels[i],
                    self.skip_channels[i] * self.pool_channel_increase_ratio[i],
                    self.skip_channels[i + 1],
                    self.skip_size[i],
                    self.skip_size[i + 1],
                    pool_stride=self.pool_strides[i],
                    kernel_size=self.conv_kernel_size[i],
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand=self.expand,
                )
            )
        return nn.ModuleList(stages)


class UBiMamba(PlainUNet):
    def __init__(
        self,
        patch_size,
        patch_channel,
        stem_channel,
        output_channel,
        n_stages,
        conv_kernel_size=3,
        pool_strides=2,
        pool_channel_increase_ratio=2,
        deep_supervision=False,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        super().__init__(
            patch_size,
            patch_channel,
            stem_channel,
            output_channel,
            n_stages,
            conv_kernel_size,
            pool_strides,
            pool_channel_increase_ratio,
            deep_supervision,
        )

    def _build_encoder(self):
        return MEncoder(
            self.patch_size,
            self.patch_channel,
            self.stem_channel,
            self.n_stages,
            self.skip_channels,
            self.skip_size,
            self.conv_kernel_size,
            self.pool_strides,
            self.pool_channel_increase_ratio,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )
