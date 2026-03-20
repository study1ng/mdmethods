import lightning as L
import torch, einops.layers.torch
from torch import nn, tensor, Tensor
import torch.nn.functional as F
from functools import partial
from experiments.nets import UNet, UNetHead
from experiments.config import image_key
from experiments.nets.plainunet import SingleConvBlock
from math import prod
from experiments.utils import assert_divisable, assert_eq


def pixelshuffle(dim: int, *scales: int) -> einops.layers.torch.Rearrange:
    assert_eq(dim, len(scales))
    scale_characters = [f"p{i}" for i in range(dim)]
    scale_pattern = " ".join(scale_characters)
    dimensions = [f"d{i}" for i in range(dim)]
    dimensions_pattern = " ".join(dimensions)
    scale_dims = [f"({scale_characters[i]} {dimensions[i]})" for i in range(dim)]
    scale_dims_pattern = " ".join(scale_dims)

    scale_dict = {scale_characters[i]: scales[i] for i in range(dim)}

    pattern = f"b ({scale_pattern} c) {dimensions_pattern} -> b c {scale_dims_pattern}"
    return einops.layers.torch.Rearrange(pattern, **scale_dict)


def generate_mask(
    image: torch.Tensor,
    mask_size: tuple[int, int, int],
    mask_ratio: float,
    device: str | None = None,
) -> torch.Tensor:
    sh = image.shape
    assert len(sh) == 5
    assert sh[2] % mask_size[0] == 0
    assert sh[3] % mask_size[1] == 0
    assert sh[4] % mask_size[2] == 0
    msh = (sh[0], 1, *[sh[i + 2] // mask_size[i] for i in range(3)])
    mask = torch.rand(msh, device=device) < mask_ratio
    mask = F.interpolate(mask.float(), sh[2:], mode="nearest")
    return mask


class RevertResolutionHead(UNetHead):
    @classmethod
    def attach_to_unet(cls, unet: UNet) -> UNet:
        if unet.deep_supervision:
            head = nn.ModuleList(
                tuple(
                    cls(
                        input_size=skip_size,
                        input_channel=skip_channel,
                        output_size=unet.patch_size,
                        output_channel=unet.output_channel,
                    )
                    for skip_size, skip_channel in zip(
                        unet.skip_size, unet.skip_channels, strict=True
                    )
                )
            )
        else:
            head = cls(
                input_size=unet.skip_size[0],
                input_channel=unet.skip_channels[0],
                output_size=unet.patch_size,
                output_channel=unet.output_channel,
            )
        unet.decoder.head = head
        return unet


class PixelShuffleHead(RevertResolutionHead):
    """PixelShuffleHeadは入力特徴マップに対してピクセルシャッフルを行うことによって期待する解像度, チャネル数に戻すヘッドである"""

    def __init__(self, input_size, input_channel, output_size, output_channel):
        super().__init__(input_size, input_channel, output_size, output_channel)
        scales = assert_divisable(output_size, input_size)
        self.ps = nn.Sequential(
            SingleConvBlock(
                input_channel, output_channel * prod(scales), input_size, kernel_size=1
            ),
            pixelshuffle(self.dim, *scales),
        )

    def _forward(self, x):
        return self.ps(x)


class MIMModule(L.LightningModule):
    def __init__(
        self,
        unet: UNet,
        mask_ratio: float,
        mask_fn=generate_mask,
        mask_size: tuple[int, ...] | None = None,
        weights: float | tuple[float, ...] | None = None,
        loss_fn=partial(nn.L1Loss, reduction="none"),
        visible_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["unet"])
        self.unet = PixelShuffleHead.attach_to_unet(unet)
        print(self.unet)
        self.deep_supervision = unet.deep_supervision
        self.mask_ratio = mask_ratio
        self.mask_fn = mask_fn
        self.mask_size = mask_size
        self.weights = weights
        self.visible_loss_weight = visible_loss_weight
        self.loss = loss_fn()

        if mask_size is None:
            self.mask_size = assert_divisable(
                self.unet.patch_size, self.unet.skip_size[-1]
            )
            print("default mask size: ", self.mask_size)
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
        self.optim = torch.optim.AdamW(
            self.unet.parameters(),
            lr=4e-4,
            eps=1e-5,
            weight_decay=1e-1,
            betas=(0.9, 0.95),
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = total_steps // 10
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optim,
            [
                torch.optim.lr_scheduler.LinearLR(
                    self.optim,
                    start_factor=1e-10,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optim, T_max=total_steps - warmup_steps, eta_min=1e-5
                ),
            ],
            milestones=[warmup_steps],
        )
        return {
            "optimizer": self.optim,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, _):
        image = batch[image_key] # (B,C,H,W,D)
        mask = self.mask_fn(image, self.mask_size, self.mask_ratio, self.device) #(B,C,H,W,D)
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
        l = loss.mean() * self.visible_loss_weight + l * (1 - self.visible_loss_weight)
        self.log("training_loss", l, prog_bar=True, on_epoch=True)
        return l
