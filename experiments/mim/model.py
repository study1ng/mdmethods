from fractions import Fraction
from logging import warning
import math
import lightning as L
import torch, einops.layers.torch
from torch import nn, tensor, Tensor
import torch.nn.functional as F
from functools import partial
from experiments.nets.base import Pool, UNetHead
from experiments.config import image_key
from experiments.nets.generic_modules import ConvBlock
from math import prod
from experiments.nets.plainunet import PlainUNet
from experiments.utils import (
    assert_divisible,
    assert_eq,
    assert_integer_scale,
    assert_to_integer,
    denominator,
    elementwise_mul,
    is_integer,
    numerator,
    reciprocal,
    to_fraction,
    repeat,
)


class RearrangePool(Pool):
    def __init__(self, input_channel, pool_scale: Fraction | tuple[Fraction, ...]):
        expand_scale = numerator(pool_scale)
        reciprocal_shrink_scale = denominator(pool_scale)
        output_channel = assert_divisible(input_channel, prod(expand_scale)) * prod(
            reciprocal_shrink_scale
        )
        super().__init__(input_channel, output_channel, pool_scale)
        self.module = self.rearrange(pool_scale)

    @staticmethod
    def rearrange(pool_scale: Fraction | tuple[Fraction, ...]):
        if isinstance(pool_scale):
            pool_scale = (pool_scale,)
        dim = len(pool_scale)
        expand_scale = numerator(pool_scale)
        reciprocal_shrink_scale = denominator(pool_scale)
        # b (c expand_scale) *(h shrink_scale) -> b (c shrink_scale) *(h expand_scale)
        expand_scale_characters = tuple(f"e{i}" for i in range(dim))
        shrink_scale_characters = tuple(f"s{i}" for i in range(dim))
        spatial_characters = tuple(f"h{i}" for i in range(dim))
        spatial_reprs_former = tuple(
            f"({spatial_characters[i]} {shrink_scale_characters[i]})"
            for i in range(dim)
        )
        spatial_reprs_later = tuple(
            f"({spatial_characters[i]} {expand_scale_characters[i]})"
            for i in range(dim)
        )
        expand_scale_dic = {
            expand_scale_characters[i]: expand_scale[i] for i in range(dim)
        }
        shrink_scale_dic = {
            shrink_scale_characters[i]: reciprocal_shrink_scale[i] for i in range(dim)
        }
        pattern_former = f'b (c {" ".join(expand_scale_characters)}) {" ".join(spatial_reprs_former)}'
        pattern_later = (
            f'b (c {" ".join(shrink_scale_characters)}) {" ".join(spatial_reprs_later)}'
        )
        return einops.layers.torch.Rearrange(
            f"{pattern_former} -> {pattern_later}",
            **expand_scale_dic,
            **shrink_scale_dic,
        )

    def _forward(self, x):
        return self.module(x)


class PixelShufflePool(Pool):
    def __init__(
        self,
        input_channel,
        output_channel,
        pool_scale: Fraction | tuple[Fraction, ...],
        *,
        conv_position: float = 1.0,  # -1.: conv -> rearrange, -0.: rearrange(shrink) -> conv -> rearrange(expand), 0.: rearrange(expand) -> conv -> rearrange(shrink), 1.: rearrange -> conv
        dim: int,
    ):
        self.conv_position = conv_position
        pool_scale = repeat(pool_scale, dim=dim, types=Fraction)
        super().__init__(input_channel, output_channel, pool_scale)
        if conv_position == -1.0:
            self.module = self._conv_first()
        elif conv_position == -0.0 and math.copysign(1, conv_position) == -1.0:
            self.module = self._shrink_first()
        elif conv_position == 0.0:
            self.module = self._expand_first()
        elif conv_position == 1.0:
            self.module = self._conv_last()
        else:
            raise ValueError(
                f"conv_position need to be in -1.0, -0.0, 0.0, 1.0, not {conv_position}"
            )

    def _conv_first(self):
        # conv_channel / pool_scale = output_channel
        conv_channel = assert_to_integer(self.output_channel * prod(self.pool_scale))
        return nn.Sequential(
            ConvBlock(
                input_channel=self.input_channel,
                output_channel=conv_channel,
                kernel_size=1,
                dim=self.dim,
            ),
            RearrangePool(input_channel=conv_channel, pool_scale=self.pool_scale),
        )

    def _conv_last(self):
        # input_channel / pool_scale = conv_channel
        conv_channel = assert_to_integer(
            self.input_channel * prod(reciprocal(self.pool_scale))
        )
        return nn.Sequential(
            RearrangePool(input_channel=self.input_channel, pool_scale=self.pool_scale),
            ConvBlock(
                input_channel=conv_channel,
                output_channel=self.output_channel,
                kernel_size=1,
                dim=self.dim,
            ),
        )

    def _shrink_first(self):
        expand_scale = to_fraction(numerator(self.pool_scale))
        shrink_scale = reciprocal(denominator(self.pool_scale))
        # input_channel / shrink_scale = before_conv_channel
        # after_conv_channel / expand_scale = output_channel
        before_conv_channel = assert_to_integer(
            self.input_channel * prod(reciprocal(shrink_scale))
        )
        after_conv_channel = assert_to_integer(
            self.output_channel * prod(expand_scale)
        )
        return nn.Sequential(
            RearrangePool(input_channel=self.input_channel, pool_scale=shrink_scale),
            ConvBlock(
                input_channel=before_conv_channel,
                output_channel=after_conv_channel,
                kernel_size=1,
                dim=self.dim,
            ),
            RearrangePool(input_channel=after_conv_channel, pool_scale=expand_scale),
        )

    def _expand_first(self):
        expand_scale = to_fraction(numerator(self.pool_scale))
        shrink_scale = reciprocal(denominator(self.pool_scale))
        # input_channel / expand_scale = before_conv_channel
        # after_conv_channel / shrink_scale = output_channel
        before_conv_channel = assert_to_integer(
            self.input_channel * prod(reciprocal(expand_scale))
        )
        after_conv_channel = assert_to_integer(
            self.output_channel * prod(shrink_scale)
        )
        return nn.Sequential(
            RearrangePool(input_channel=self.input_channel, pool_scale=expand_scale),
            ConvBlock(
                input_channel=before_conv_channel,
                output_channel=after_conv_channel,
                kernel_size=1,
                dim=self.dim,
            ),
            RearrangePool(input_channel=after_conv_channel, pool_scale=shrink_scale),
        )

    def _forward(self, x):
        return self.module(x)


class MaskGenerator(nn.Module):
    def __init__(
        self,
        mask_scale: Fraction | tuple[Fraction, ...],
        mask_ratio: float,
        *,
        dim: int,
    ):
        super().__init__()
        self.dim = dim
        self.mask_scale = to_fraction(mask_scale)
        self.mask_scale = repeat(mask_scale, dim, types=Fraction)
        self.mask_ratio = mask_ratio

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *s = x.shape
        assert len(s) == self.dim
        mask_size = assert_integer_scale(s, self.mask_scale)
        msh = (b, 1, *mask_size)
        mask = torch.rand(msh, device=x.device) < self.mask_ratio
        mask = F.interpolate(mask.float(), s, mode="nearest")
        return mask


class RevertResolutionHead(UNetHead):
    """A head which output size is as same as unet patch size."""

    def __init__(
        self, input_channel, output_channel, pool_scale: Fraction, *, dim: int
    ):
        self.dim = dim
        self.output_channel = output_channel
        self.pool_scale = pool_scale
        self.pool_scale = repeat(self.pool_scale, self.dim, types=Fraction)
        super().__init__(input_channel)

    @classmethod
    def attach_to_unet(
        cls,
        unet: PlainUNet,
        output_scale: Fraction | tuple[Fraction, ...] = Fraction(
            1,
        ),
        *args,
        **kwargs,
    ) -> PlainUNet:
        """attach myself to unet

        Parameters
        ----------
        unet : PlainUNet
            UNet object

        output_scale : Fraction

        Returns
        -------
        UNet
            attached UNet
        """
        output_scale = repeat(output_scale, unet.dim, types=Fraction)
        if not unet.deep_supervision:
            head = cls(
                *args,
                input_channel=unet.skip_channels[0],
                output_channel=unet.output_channel,
                pool_scale=output_scale,
                dim=unet.dim,
                **kwargs,
            )
            unet.decoder.head = head
            return unet

        head_scales = [output_scale]
        for scale in unet.decoder_pool_scales:
            output_scale = elementwise_mul(output_scale, scale)
            head_scales.append(output_scale)

        heads = nn.ModuleList(
            tuple(
                cls(
                    *args,
                    input_channel=input_channel,
                    output_channel=unet.output_channel,
                    pool_scale=scale,
                    dim=unet.dim,
                    **kwargs,
                )
                for input_channel, scale in zip(
                    unet.skip_channels, head_scales, strict=True
                )
            )
        )
        unet.decoder.head = heads
        return unet


class PixelShuffleHead(RevertResolutionHead):
    """PixelShuffleHeadは入力特徴マップに対してピクセルシャッフルを行うことによって期待する解像度, チャネル数に戻すヘッドである"""

    def __init__(
        self,
        input_channel,
        output_channel,
        scale: Fraction | tuple[Fraction, ...],
        *,
        dim: int,
        conv_position: float = 1.0,
    ):
        self.output_channel = output_channel
        self.conv_position = conv_position
        super().__init__(input_channel, output_channel, pool_scale=scale, dim=dim)
        self.module = PixelShufflePool(
            input_channel=self.input_channel,
            output_channel=self.output_channel,
            pool_scale=self.pool_scale,
            conv_position=self.conv_position,
            dim=self.dim,
        )

    def calculate_output_size(self, input_size):
        return self.module.calculate_output_size(input_size)
    
    def _forward(self, x):
        return self.module(x)


class MIMModule(L.LightningModule):
    """A module which does Masking Image Modeling"""

    def __init__(
        self,
        unet: PlainUNet,
        mask_ratio: float,
        mask_gen=MaskGenerator,
        head=PixelShuffleHead,
        mask_scale: Fraction | tuple[Fraction, ...] | None = None,
        weights: float | tuple[float, ...] | None = None,
        loss_fn=partial(nn.L1Loss, reduction="none"),
        visible_loss_weight: float = 0.0,
        *,
        conv_position: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["unet"])
        self.head = head
        self.unet = head.attach_to_unet(unet, conv_position=conv_position)
        print(self.unet)
        self.deep_supervision = unet.deep_supervision
        self.mask_ratio = mask_ratio
        self.scale = mask_scale
        self.weights = weights
        self.visible_loss_weight = visible_loss_weight
        self.loss = loss_fn()

        if mask_scale is None:
            mask_scale = repeat(
                Fraction(
                    1,
                ),
                dim=self.unet.dim,
            )
            for scale in self.unet.encoder_pool_scales:
                mask_scale = elementwise_mul(mask_scale, scale)
            self.mask_scale = mask_scale
            print("default mask scale: ", self.mask_scale)
        self.mask_fn = mask_gen(self.mask_scale, self.mask_ratio, dim=self.unet.dim)
        if self.deep_supervision:
            if isinstance(self.weights, tuple):
                assert_eq(len(self.unet.decoder.head), len(self.weights))
            if self.weights is None:
                self.weights = 0.5
            if isinstance(self.weights, float):
                self.weights = tuple(
                    self.weights**i for i in range(self.unet.n_stages + 1)
                )
            self.weights = tensor(self.weights)
            self.weights /= self.weights.sum()
            self.register_buffer("head_weights", self.weights)

    def forward(self, x: Tensor):
        return self.unet(x)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.unet.parameters(),
            lr=4e-4,
            eps=1e-5,
            weight_decay=1e-1,
            betas=(0.9, 0.95),
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = total_steps // 10
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optim,
            [
                torch.optim.lr_scheduler.LinearLR(
                    optim,
                    start_factor=1e-10,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim, T_max=total_steps - warmup_steps, eta_min=1e-5
                ),
            ],
            milestones=[warmup_steps],
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, _):
        image = batch[image_key]  # (B,C,H,W,D)
        mask_size = elementwise_mul(image.shape[2:], self.mask_scale)
        if not all(is_integer(mask_size)):
            warning(
                f"patch shape is {image.shape}, which product with mask_scale {self.mask_scale} is not integer"
            )
        mask = self.mask_fn(image)  # (B,C,H,W,D)
        # masked = image * (1 - mask) + self.mask_token * mask
        masked = image * (1 - mask)

        out = self.unet(masked)
        if self.deep_supervision:
            loss = 0
            for i in range(len(out)):
                loss += self.head_weights[i] * self.loss(out[i], image)
        else:
            loss = self.loss(out, image)
        l = loss * mask
        l = l.sum() / (mask.sum() * image.shape[1] + 1e-5)
        vl = loss * (1 - mask)
        vl = vl.sum() / ((1 - mask).sum() * image.shape[1] + 1e-5)  # visible loss
        l = vl * self.visible_loss_weight + l * (1 - self.visible_loss_weight)
        self.log("training_loss", l, prog_bar=True, on_epoch=True)
        return l
