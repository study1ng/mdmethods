from torch import Tensor, nn, tensor
import torch
from experiments.utils import (
    assert_divisable,
    get_gaussian_kernel,
    size_of_tensor,
    assert_eq,
)
import math
import einops
import torch.nn.functional as F
from experiments.mim.model import generate_mask, pixelshuffle
from math import prod
from experiments.nets.plainunet import SingleConvBlock
from experiments.config import image_key
import lightning as L
import torch
from experiments.nets import UNet, UNetHead


class HogLayer3D(nn.Module):
    """正12面体の20頂点を利用したHOG特徴量の計算"""

    def __init__(
        self,
        cell_size: int = 7,
        block_size: int = 2,
        gaussian_window_size: tuple[int, int, int] | None = None,
        signed: bool = True,
    ):
        super().__init__()
        self.cell_size = cell_size
        self.block_size = block_size
        self.gaussian_window_size = gaussian_window_size
        self.signed = signed
        if self.signed:
            self.bin_count = 20
        else:
            self.bin_count = 10
        if gaussian_window_size is not None:
            gaussian_window = get_gaussian_kernel(
                self.gaussian_window_size,
                tuple(s // 2 for s in self.gaussian_window_size),
                dim=3,
            )
            self.register_buffer("gaussian_window", gaussian_window)

        phi = (1.0 + math.sqrt(5.0)) / 2.0
        rphi = 2.0 / (1.0 + math.sqrt(5.0))
        vertexes = [
            [1, 1, 1],
            [-1, -1, -1],
            [1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [-1, 1, -1],
            [1, -1, -1],
            [-1, 1, 1],
            [0, rphi, phi],
            [0, -rphi, -phi],
            [0, rphi, -phi],
            [0, -rphi, phi],
            [phi, 0, rphi],
            [-phi, 0, -rphi],
            [phi, 0, -rphi],
            [-phi, 0, rphi],
            [rphi, phi, 0],
            [-rphi, -phi, 0],
            [rphi, -phi, 0],
            [-rphi, phi, 0],
        ]
        vertexes = torch.tensor(vertexes, dtype=torch.float).T
        self.register_buffer("vertexes", vertexes, persistent=False)
        differential = torch.tensor([-1, 0, 1], dtype=torch.float)
        g = torch.outer(
            torch.tensor([1, 2, 1], dtype=torch.float),
            torch.tensor([1, 2, 1], dtype=torch.float),
        )
        sobel_d = g[..., None] * differential  # depth
        sobel_d = sobel_d.view(1, 1, 3, 3, 3)
        sobel_w = einops.rearrange(sobel_d, "b c h w d->b c h d w")  # width
        sobel_h = einops.rearrange(sobel_d, "b c h w d->b c d w h")  # height
        self.register_buffer("weight_d", sobel_d, persistent=False)
        self.register_buffer("weight_w", sobel_w, persistent=False)
        self.register_buffer("weight_h", sobel_h, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w, d = x.shape

        def _pad(i):
            return (self.cell_size - i % self.cell_size) % self.cell_size

        # xをcell_sizeの倍数になるように調整
        pad_h = _pad(h)
        pad_w = _pad(w)
        pad_d = _pad(d)
        x = F.pad(
            x, pad=[0, pad_d, 0, pad_w, 0, pad_h], mode="reflect"
        )  # x: (B, C, H, W, D)
        b, c, h, w, d = x.shape
        x = F.pad(x, pad=[1, 1, 1, 1, 1, 1], mode="reflect")
        wh = self.weight_h.repeat(c, 1, 1, 1, 1)
        ww = self.weight_w.repeat(c, 1, 1, 1, 1)
        wd = self.weight_d.repeat(c, 1, 1, 1, 1)
        gh = F.conv3d(x, wh, bias=None, stride=1, padding=0, groups=c)
        gw = F.conv3d(x, ww, bias=None, stride=1, padding=0, groups=c)
        gd = F.conv3d(x, wd, bias=None, stride=1, padding=0, groups=c)
        ghwd = torch.stack((gh, gw, gd), dim=-1)  # (B, C, H, W, D, 3)
        norm = ghwd.norm(dim=-1)
        phase = (
            ghwd @ self.vertexes
        )  # inner product (B,C,H,W,D,3)@(3,20)->(B,C,H,W,D,20)
        if not self.signed:
            phase = phase.abs().view(-1, 2).sum(dim=-1)  # (B,C,H,W,D,bin_count)
        bn = torch.argmax(phase, dim=-1)  # (B,C,H,W,D)
        if self.gaussian_window_size:
            repeat_rate = assert_divisable(size_of_tensor(x), self.gaussian_window_size)
            temp_gkern = self.gaussian_window.repeat(repeat_rate)
            norm *= temp_gkern
        bn = (
            bn.unfold(2, self.cell_size, self.cell_size)
            .unfold(3, self.cell_size, self.cell_size)
            .unfold(4, self.cell_size, self.cell_size)
        )
        norm = (
            norm.unfold(2, self.cell_size, self.cell_size)
            .unfold(3, self.cell_size, self.cell_size)
            .unfold(4, self.cell_size, self.cell_size)
        )

        bn = bn.flatten(
            start_dim=-3
        )  # (B, C, H//self.cell_size, W//self.cell_size, D//self.cell_size, self.cell_size^3)
        norm = norm.flatten(
            start_dim=-3
        )  # (B, C, H//self.cell_size, W//self.cell_size, D//self.cell_size, self.cell_size^3)

        h_out, w_out, d_out = assert_divisable((h, w, d), self.cell_size)
        out = torch.zeros(
            (b, c, h_out, w_out, d_out, self.bin_count),
            dtype=torch.float,
            device=x.device,
        )  # (B, C, H // self.cell_size, W // self.cell_size, D // self.cell_size, ,bin_count)

        out.scatter_add_(dim=-1, index=bn, src=norm)
        out = (
            out.unfold(2, self.block_size, 1)
            .unfold(3, self.block_size, 1)
            .unfold(4, self.block_size, 1)
        )
        out = out.flatten(start_dim=5)
        out = F.normalize(out, p=2, dim=-1, eps=1e-5)
        return out

    def output_size(self, input_size: tuple[int, ...]) -> tuple[int, ...]:
        assert_eq(5, len(input_size))
        b, c, h, w, d = input_size

        def _div_ceiling(i):
            return (i + self.cell_size - 1) // self.cell_size

        h = _div_ceiling(h)
        w = _div_ceiling(w)
        d = _div_ceiling(d)
        h = h - self.block_size + 1
        w = w - self.block_size + 1
        d = d - self.block_size + 1
        out_features = self.bin_count * (self.block_size**3)
        return (b, c, h, w, d, out_features)


class HogHead(UNetHead):
    def __init__(
        self, input_size, input_channel, output_size, output_channel, hog_channel: int
    ):
        super().__init__(input_size, input_channel, output_size, output_channel)
        scales = assert_divisable(output_size, input_size) + (hog_channel,)
        self.ps = nn.Sequential(
            SingleConvBlock(
                input_channel, output_channel * prod(scales), input_size, kernel_size=1
            ),
            pixelshuffle(self.dim + 1, *scales),
        )

    def _forward(self, x):
        return self.ps(x)

    @classmethod
    def attach_to_unet(cls, unet: UNet, hog: HogLayer3D):
        unet_output_shape = (1, unet.output_channel, *unet.patch_size)
        output_shape = hog.output_size(unet_output_shape)
        output_channel = output_shape[1]
        output_size = output_shape[2:5]
        hog_channel = output_shape[5]

        if unet.deep_supervision:
            head = nn.ModuleList(
                tuple(
                    cls(
                        input_size=skip_size,
                        input_channel=skip_channel,
                        output_size=output_size,
                        output_channel=output_channel,
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
                output_size=output_size,
                output_channel=output_channel,
                hog_channel=hog_channel,
            )
        unet.decoder.head = head
        return unet


class MaskFeatModule(L.LightningModule):
    def __init__(
        self,
        unet: UNet,
        mask_ratio: float,
        mask_fn=generate_mask,
        mask_size: tuple[int, ...] | None = None,
        weights: float | tuple[float, ...] | None = None,
        loss_fn=nn.MSELoss,
        cell_size: int = 7,
        block_size: int = 2,
        gaussian_window_size=None,
        signed: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["unet"])
        self.hog = HogLayer3D(
            cell_size=cell_size,
            block_size=block_size,
            gaussian_window_size=gaussian_window_size,
            signed=signed,
        )
        self.unet = HogHead.attach_to_unet(unet, self.hog)
        print(self.unet)
        self.deep_supervision = unet.deep_supervision
        self.mask_ratio = mask_ratio
        self.mask_fn = mask_fn
        self.mask_size = mask_size
        self.weights = weights
        self.loss = loss_fn()

        if mask_size is None:
            self.mask_size = assert_divisable(
                self.unet.patch_size, self.unet.skip_size[-1]
            )
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
        image = batch[image_key]
        hog = self.hog(image)
        mask = self.mask_fn(image, self.mask_size, self.mask_ratio, self.device)
        # masked = image * (1 - mask) + self.mask_token * mask
        masked = image * (1 - mask)

        out = self.unet(masked)
        if self.deep_supervision:
            loss = 0
            for i in range(len(out)):
                loss += self.head_weights[i] * self.loss(out[i], hog)
        else:
            loss = self.loss(out, hog)
        self.log("training_loss", loss, prog_bar=True, on_epoch=True)
        return loss
