import lightning as L
import torch, math, einops.layers.torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from functools import partial
from methods.structures.nets.UBiMambaEnc_3d import UNet


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


class UNetEncoder(nn.Module):
    def __init__(self, unet: UNet):
        super().__init__()
        assert hasattr(
            unet, "encoder"
        ), "UNetEncoder needs a unet object which has 'encoder' attribute"
        assert hasattr(
            unet, "input_channels"
        ), "UNetEncoder needs a unet object which has 'input_channels' attribute"
        assert hasattr(
            unet, "input_size"
        ), "UNetEncoder needs a unet object which has 'input_size' attribute"
        self.encoder = unet.encoder
        self.input_channel = unet.input_channels
        self.input_shape = unet.input_size
        self.num_stages = unet.n_stages
        self.output_shapes = unet.output_size
        self.output_channels = unet.output_channels

    def forward(self, x):
        return self.encoder(x)


class UNetEncoderHead(nn.Module, ABC):
    @abstractmethod
    def __init__(
        self,
        channels: list,
        shapes: list,
        patch_channel: int,
        patch_shape: list,
        deep_supervision: bool = False,
    ): ...

    @abstractmethod
    def forward(self, x): ...


class PixelShuffle(UNetEncoderHead, nn.Module):
    def __init__(
        self,
        channels: list,
        shapes: list,
        patch_channel: int,
        patch_shape: list,
        deep_supervision: bool = False,
    ):
        # loss_fn: __init__(channel: int, shape: sequence[int]), forward(predicted, teacher)の
        # 二つのメソッドを持つオブジェクトを返す関数
        nn.Module.__init__(self)
        self.deep_supervision = deep_supervision
        if self.deep_supervision:
            assert len(channels) == len(
                shapes
            ), f"deep supervised loss with unmatched channels and shapes"
            self.head = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv3d(
                            channel,
                            patch_channel
                            * math.prod(
                                patch_shape[i] // shape[i] for i in range(len(shape))
                            ),
                            1,
                        ),
                        einops.layers.torch.Rearrange(
                            "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
                            p1=patch_shape[0] // shape[0],
                            p2=patch_shape[1] // shape[1],
                            p3=patch_shape[2] // shape[2],
                        ),
                    )
                    for channel, shape in zip(channels, shapes)
                ]
            )
        else:
            channel = channels[-1]
            shape = shapes[-1]
            self.head = nn.Sequential(
                nn.Conv3d(
                    channel,
                    patch_channel
                    * math.prod(patch_shape[i] // shape[i] for i in range(len(shape))),
                    1,
                ),
                einops.layers.torch.Rearrange(
                    "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
                    p1=patch_shape[0] // shape[0],
                    p2=patch_shape[1] // shape[1],
                    p3=patch_shape[2] // shape[2],
                ),
            )

    def forward(self, skips):
        if self.deep_supervision:
            out = [head(skip) for head, skip in zip(self.head, skips, strict=True)]
            return out
        else:
            out = self.head(skips[-1])
            return out


class UNetEncoderWithHead(UNetEncoder):
    def __init__(
        self, unet, head_fn, patch_channel, patch_shape, deep_supervision: bool = False
    ):
        super().__init__(unet)
        self.head = head_fn(
            self.output_channels,
            self.output_shapes,
            patch_channel,
            patch_shape,
            deep_supervision,
        )

    def forward(self, x):
        return self.head(super().forward(x))


class UNetEncoderWithMIM(L.LightningModule):
    def __init__(
        self,
        unet: UNet,
        patch_channel,
        patch_shape,
        mask_ratio,
        mask_fn=generate_mask,
        loss_fn=partial(nn.L1Loss, reduction="none"),
        mask_shape=None,
        deep_supervision: bool = False,
        weights: float | list[float] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="unet")
        self.encoder = UNetEncoderWithHead(
            unet, PixelShuffle, patch_channel, patch_shape, deep_supervision
        )
        print(self.encoder)
        shape = self.encoder.output_shapes[-1]
        if mask_shape is None:
            self.mask_shape = [
                patch_shape[i] // shape[i] for i in range(len(patch_shape))
            ]
        else:
            self.mask_shape = mask_shape
        self.mask_fn = mask_fn
        self.mask_ratio = mask_ratio
        self.deep_supervision = deep_supervision
        self.patch_channel = patch_channel
        self.patch_shape = patch_shape
        self.loss = loss_fn()
        self.weights = weights
        if self.weights is None:
            self.weights = [2.**i for i in range(self.encoder.num_stages)]
        elif isinstance(self.weights, float):
            self.weights = [self.weights**i for i in range(self.encoder.num_stages)]
        self.weights = np.array(self.weights)
        self.weights /= self.weights.sum()
        self.weights = torch.Tensor(self.weights)
        self.register_buffer("loss_weights", self.weights, persistent=True)
        # self.mask_token = nn.Parameter(
        #     torch.zeros(1, self.encoder.input_channel, 1, 1, 1)
        # )

    def forward(self, x):
        out = self.encoder(x)
        return out

    def configure_optimizers(self):
        self.optim = torch.optim.AdamW(
            self.encoder.parameters(),
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
        image = batch["image"]
        mask = generate_mask(image, self.mask_shape, self.mask_ratio, self.device)
        # masked = image * (1 - mask) + self.mask_token * mask
        masked = image * (1 - mask)

        out = self.encoder(masked)
        if self.deep_supervision:
            loss = 0
            for i in range(len(out)):
                loss += self.weights[i] * self.loss(out[i], image)
        else:
            loss = self.loss(out, image)
        l = loss * mask
        l = l.sum() / (mask.sum() * image.shape[1] + 1e-5)
        self.log("training_loss", l, prog_bar=True, on_epoch=True)
        return l

    @classmethod
    def from_plan(
        cls,
        plan,
        unetclass,
        mask_ratio,
        loss_fn=partial(nn.L1Loss, reduction="none"),
        mask_fn=generate_mask,
        mask_shape=None,
        deep_supervision: bool = False,
        weights: list[float] | None = None,
    ):
        return cls(
            unetclass.from_plan(plan, 1, 1),
            1,
            plan["configurations"]["3d_fullres"]["patch_size"],
            mask_ratio,
            mask_fn,
            loss_fn,
            mask_shape,
            deep_supervision,
            weights,
        )
