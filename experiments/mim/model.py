import lightning as L
import torch, einops.layers.torch
from torch import nn, tensor, Tensor
import torch.nn.functional as F
from functools import partial
from experiments.nets.base import UNet, UNetHead
from experiments.config import image_key
from experiments.nets.generic_modules import ConvBlock
from math import prod
from experiments.nets.plainunet import PlainUNet
from experiments.utils import (
    assert_divisible,
    assert_eq,
    elementwise_gt,
    elementwise_le,
    elementwise_mul,
    size_of_tensor,
)


def pixelshuffle(
    dim: int, *scales: int, shrink: bool = False
) -> einops.layers.torch.Rearrange:
    assert_eq(dim, len(scales))
    scale_characters = [f"p{i}" for i in range(dim)]
    scale_pattern = " ".join(scale_characters)
    dimensions = [f"d{i}" for i in range(dim)]
    dimensions_pattern = " ".join(dimensions)
    scale_dims = [f"({dimensions[i]} {scale_characters[i]})" for i in range(dim)]
    scale_dims_pattern = " ".join(scale_dims)
    scale_dict = {scale_characters[i]: scales[i] for i in range(dim)}
    if shrink:
        pattern = (
            f"b c {scale_dims_pattern} -> b (c {scale_pattern}) {dimensions_pattern}"
        )
    else:
        pattern = (
            f"b (c {scale_pattern}) {dimensions_pattern} -> b c {scale_dims_pattern}"
        )
    return einops.layers.torch.Rearrange(pattern, **scale_dict)


class MaskGenerator(nn.Module):
    def __init__(
        self,
        mask_size: tuple[int, int, int],
        mask_ratio: float,
    ):
        super().__init__()
        self.mask_size = mask_size
        self.mask_ratio = mask_ratio

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sh = x.shape
        assert len(sh) == 5
        scales = assert_divisible(size_of_tensor(x), self.mask_size)
        msh = (sh[0], 1, *scales)
        mask = torch.rand(msh, device=x.device) < self.mask_ratio
        mask = F.interpolate(mask.float(), sh[2:], mode="nearest")
        return mask


class RevertResolutionHead(UNetHead):
    """A head which output size is as same as unet patch size."""

    @classmethod
    def attach_to_unet(
        cls, unet: PlainUNet, initial_scale: int = 1, shrink=False
    ) -> UNet:
        """attach myself to unet

        Parameters
        ----------
        unet : PlainUNet
            UNet object
        initial_scale : int, optional
            the target scale compared to unet output size, if you want to shrink it, check shrink argument, by default 1
        shrink : bool, optional
            whether to shrink, by default False

        Returns
        -------
        UNet
            attached UNet
        """
        if not unet.deep_supervision:
            head = cls(
                input_channel=unet.skip_channels[0],
                output_channel=unet.output_channel,
                scale=initial_scale,
                shrink=shrink,
            )
            unet.decoder.head = head
            return unet
        
        def _is_shrink(i, j):
            """if it is shrinking from i to j"""
            if all(elementwise_le(i, j)):
                return False # expanding
            if all(elementwise_gt(i, j)):
                return True # shrinking
            raise NotImplementedError("some elements is shrinking and some elements is expanding.")

        former = 1
        scales_to_output = [former]
        for scale in unet.pool_strides:
            former = elementwise_mul(former, scale)
            scales_to_output.append(former)
        
        scales_to_initial_scale = []
        for scale in scales_to_output:
            if _is_shrink(scale, initial_scale):
                scales_to_initial_scale.append((assert_divisible(scale, initial_scale), True))
            else:
                scales_to_initial_scale.append((assert_divisible(initial_scale, scale), False))

        heads = []
        for i in range(unet.n_stages + 1):
            input_channel = unet.skip_channels[i]
            output_channel = unet.output_channel
            scale, shrink = scales_to_initial_scale[i]
            heads.append(cls(input_channel=input_channel, output_channel=output_channel, scale=scale, shrink=shrink))

        heads = nn.ModuleList(heads)
        unet.decoder.head = heads
        return unet

class PixelShuffleHead(RevertResolutionHead):
    """PixelShuffleHeadは入力特徴マップに対してピクセルシャッフルを行うことによって期待する解像度, チャネル数に戻すヘッドである"""

    def __init__(
        self, input_channel, output_channel, scales, *, shrink=False, dim: int
    ):
        self.output_channel = output_channel
        self.scale = scales
        self.shrink = shrink
        self.dim = dim
        super().__init__(input_channel)
        ps = pixelshuffle(self.dim, *scales, shrink=self.shrink)
        self.ps = (
            nn.Sequential(
                ConvBlock(
                    input_channel, output_channel * prod(scales), kernel_size=1, dim=dim
                ),
                ps,
            )
            if self.shrink
            else nn.Sequential(
                ps,
                ConvBlock(
                    assert_divisible(self.input_channel, prod(scales)),
                    output_channel,
                    kernel_size=1,
                    dim=dim,
                ),
            )
        )

    def calculate_output_size(self, input_size):
        if self.shrink:
            return assert_divisible(input_size, self.scale)
        else:
            return elementwise_mul(input_size, self.scale)


class MIMModule(L.LightningModule):
    """A module which does Masking Image Modeling"""

    def __init__(
        self,
        unet: UNet,
        mask_ratio: float,
        mask_gen=MaskGenerator,
        head_fn=PixelShuffleHead.attach_to_unet,
        mask_size: tuple[int, ...] | None = None,
        weights: float | tuple[float, ...] | None = None,
        loss_fn=partial(nn.L1Loss, reduction="none"),
        visible_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["unet"])
        self.head = head_fn
        self.unet = head_fn.attach_to_unet(unet)
        print(self.unet)
        self.deep_supervision = unet.deep_supervision
        self.mask_ratio = mask_ratio
        self.mask_size = mask_size
        self.weights = weights
        self.visible_loss_weight = visible_loss_weight
        self.loss = loss_fn()

        if mask_size is None:
            self.mask_size = assert_divisible(
                self.unet.patch_size, self.unet.skip_size[-1]
            )
            print("default mask size: ", self.mask_size)
        self.mask_fn = mask_gen(self.mask_size, self.mask_ratio)
        if self.deep_supervision:
            if isinstance(self.weights, tuple):
                assert_eq(len(self.unet.decoder.head), len(self.weights))
            if self.weights is None:
                self.weights = 2.0
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
