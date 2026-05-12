from fractions import Fraction
from functools import partial
from pprint import pprint

import einops.layers
import einops.layers.torch
from torch import Tensor, nn
import torch
from experiments.nets.builder import Builder
from experiments.nets.plainunet import PlainUNet
from experiments.trainer import UNetTrainingModule
from experiments.utils import (
    assert_divisible,
    assert_to_integer,
    elementwise_mul,
    get_gaussian_kernel,
    AssertEq,
)
from experiments.utils.wraputils import (
    element_wise2,
    reciprocal,
    repeat,
    element_wise2,
    to_fraction_with_denominator,
)
import math
import einops
import torch.nn.functional as F
from experiments.mim.model import (
    ConvPosition,
    MaskGenerator,
    PixelShuffleHead,
    RevertResolutionHead,
)
from math import prod
from experiments.config import image_key
import torch
import sympy


class HogLayer3D(nn.Module):
    """Calculate 3D HOG use Regular 12-ly hedron vertexes, allow anisotropy

    Parameters
    ----------
    cell_size : tuple[int, ...] | int = 8
        The cell size of HOG
    block_size : tuple[int, ...] | int = 2,
        The block size of HOG
    spacing : tuple[float, ...] | float = 1.0,
        The spacing of input image
    gaussian_window_size : tuple[int, int, int] | int | None = None,
        Gaussian window size. if None then would not use gaussian window
    signed : bool = True
        If use signed HOG
    """

    def __init__(
        self,
        cell_size: tuple[int, ...] | int = 8,
        block_size: tuple[int, ...] | int = 2,
        spacing: tuple[float, ...] | float = 1.0,
        gaussian_window_size: tuple[int, int, int] | int | None = None,
        signed: bool = True,
    ):
        super().__init__()
        self.cell_size = repeat(cell_size, 3, types=int)
        self.block_size = repeat(block_size, 3, types=int)
        self.spacing = repeat(spacing, 3, types=(int, float))
        self.gaussian_window_size = repeat(gaussian_window_size, 3, types=int)
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
        m = torch.diag(1.0 / torch.tensor(self.spacing, dtype=torch.float))
        vertexes = torch.tensor(vertexes, dtype=torch.float)
        vertexes = vertexes @ m
        vertexes = F.normalize(vertexes, p=2, dim=1).T
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
        """Calculate HOG

        Parameters
        ----------
        x : Tensor (B,C,H,W,D)
            image

        Returns
        -------
        Tensor (B,C,H,W,D,20 if signed else 10)
            HOG
        """
        b, c, h, w, d = x.shape

        @element_wise2(int)
        def _pad(i, cs):
            return (cs - i % cs) % cs

        # xをcell_sizeの倍数になるように調整
        pad_h, pad_w, pad_d = _pad((h, w, d), self.cell_size)
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
            new_shape = phase.shape[:-1] + (10, 2)
            phase = phase.view(new_shape).max(dim=-1)[0]  # (B,C,H,W,D,10)

        bn = torch.argmax(phase, dim=-1)  # (B,C,H,W,D)
        if self.gaussian_window_size:
            repeat_rate = assert_divisible(norm.shape[2:], self.gaussian_window_size)
            temp_gkern = self.gaussian_window.repeat(repeat_rate)
            norm *= temp_gkern
        bn = (
            bn.unfold(2, self.cell_size[0], self.cell_size[0])
            .unfold(3, self.cell_size[1], self.cell_size[1])
            .unfold(4, self.cell_size[2], self.cell_size[2])
        )
        norm = (
            norm.unfold(2, self.cell_size[0], self.cell_size[0])
            .unfold(3, self.cell_size[1], self.cell_size[1])
            .unfold(4, self.cell_size[2], self.cell_size[2])
        )

        bn = bn.flatten(
            start_dim=-3
        )  # (B, C, H//self.cell_size, W//self.cell_size, D//self.cell_size, self.cell_size^3)
        norm = norm.flatten(
            start_dim=-3
        )  # (B, C, H//self.cell_size, W//self.cell_size, D//self.cell_size, self.cell_size^3)

        h_out, w_out, d_out = assert_divisible((h, w, d), self.cell_size)
        out = torch.zeros(
            (b, c, h_out, w_out, d_out, self.bin_count),
            dtype=torch.float,
            device=x.device,
        )  # (B, C, H // self.cell_size, W // self.cell_size, D // self.cell_size, ,bin_count)

        out.scatter_add_(dim=-1, index=bn, src=norm)
        out = (
            out.unfold(2, self.block_size[0], 1)
            .unfold(3, self.block_size[1], 1)
            .unfold(4, self.block_size[2], 1)
        )
        out = out.flatten(start_dim=5)
        out = F.normalize(out, p=2, dim=-1, eps=1e-5)
        return out

    def output_size(self, input_size: tuple[int, ...]) -> tuple[int, ...]:
        """calculate the output size of this HOG

        Parameters
        ----------
        input_size : tuple[int, int, int, int, int]
            the input image size (B,C,H,W,D)

        Returns
        -------
        tuple[int, int, int, int, int, int]
            the output hog size (B,C,H,W,D,HOG feature)
        """
        AssertEq()(5, len(input_size))
        b, c, h, w, d = input_size

        @element_wise2(int)
        def _div_ceiling(i, cs):
            return (i + cs - 1) // cs

        h, w, d = _div_ceiling((h, w, d), self.cell_size)
        h = h - self.block_size[0] + 1
        w = w - self.block_size[1] + 1
        d = d - self.block_size[2] + 1
        out_features = self.bin_count * prod(self.block_size)
        return (
            b,
            c,
            h,
            w,
            d,
            out_features,
        )  # (112,112,128)->(15,15,18,160), 1605632->648000


class HogHead(RevertResolutionHead):
    def __init__(
        self,
        input_channel,
        output_channel,
        pool_scale,
        hog_channel: int,
        *,
        dim,
        conv_position: ConvPosition = ConvPosition.default(),
    ):
        super().__init__(input_channel, output_channel, pool_scale, dim=dim)
        self.hog_channel = hog_channel
        self.conv_position = conv_position
        self.module = nn.Sequential(
            PixelShuffleHead(
                self.input_channel,
                self.output_channel * self.hog_channel,
                pool_scale=self.pool_scale,
                dim=self.dim,
                conv_position=self.conv_position,
            ),
            einops.layers.torch.Rearrange(
                "b (c p) h w d -> b c h w d p", p=self.hog_channel
            ),
        )

    @classmethod
    def _reinitialize_unet(cls, unet, hog_channel: int, output_scale, *args, **kwargs):
        return super()._reinitialize_unet(
            unet=unet,
            output_scale=output_scale,
            *args,
            **kwargs,
            hog_channel=hog_channel,
        )

    def _forward(self, x):
        return self.module(x)

    def calculate_output_size(self, input_size):
        b, c, h, w, d = elementwise_mul(input_size, self.pool_scale)
        return (b, assert_divisible(c, self.hog_channel), h, w, d, self.hog_channel)


class MaskFeatModule(UNetTrainingModule):
    def __init__(
        self,
        builder: list[dict],
        mask_ratio,
        mask_gen=MaskGenerator,
        mask_scale=None,
        weights=None,
        loss_fn=partial(nn.MSELoss, reduction="none"),
        visible_loss_weight=0.0,
        *,
        conv_position=ConvPosition.default(),
        cell_size: tuple[int, ...] | int | None = None,
        gaussian_window_size: tuple[int, int, int] | int | None = None,
        signed: bool = True,
    ):
        super().save_hyperparameters()
        self.cell_size = cell_size
        self.gaussian_window_size = gaussian_window_size
        self.signed = signed
        self.conv_position = conv_position
        self.mask_ratio = mask_ratio
        self.mask_scale = mask_scale
        self.weights = weights
        self.visible_loss_weight = visible_loss_weight
        if mask_scale is None:
            print(builder)
            unet = Builder.from_params(builder).build() # unefficient build but we need to initialize mask_scale
            mask_scale = repeat(
                Fraction(
                    1,
                ),
                dim=unet.dim,
            )
            for scale in unet.encoder_pool_scales:
                mask_scale = elementwise_mul(mask_scale, scale)
            self.mask_scale = mask_scale
            print("default mask scale: ", self.mask_scale)
        mask_size = assert_to_integer(reciprocal(self.mask_scale))
        if cell_size is None:
            # セルサイズはmask_sizeの約数でありかつ, できるだけ(8,8,8)に近い値を探索する.
            @element_wise2(int)
            def _find_closest_divisor(i, j):
                """iの約数のうち, 最もjと値が近いものを返す"""
                divisors = sympy.divisors(i)
                closest = min(divisors, key=lambda x: (abs(x - j), x))
                return closest

            # mask_sizeの約数となるように調整
            self.cell_size = _find_closest_divisor(mask_size, 8)
            print("default cell size: ", self.cell_size)
        hog = HogLayer3D(
            cell_size=self.cell_size,
            block_size=1,  # ブロックでの正規化は行わない
            gaussian_window_size=self.gaussian_window_size,
            signed=self.signed,
        )
        builder = Builder.from_params(builder).reinitialize(
            "maskfeat.model.HogHead",
            hog_channel=hog.bin_count,
            output_scale=repeat(
                to_fraction_with_denominator(1, self.cell_size),
                dim=len(self.mask_scale),
                types=Fraction,
            ),
            conv_position=conv_position,
        ).to_params()
        super().__init__(builder=builder, weights=weights)

        self.cell_per_mask = assert_divisible(mask_size, self.cell_size)
        self.loss = loss_fn()
        self.hog = hog
        self.mask_fn = mask_gen(self.mask_scale, self.mask_ratio, dim=self.unet.dim)

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
        warmup_steps = min(total_steps, 1000) // 10
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
        image = batch[image_key]
        hog = self.hog(image)  # (B,C,H',W',D',bins)
        mask = self.mask_fn(image)  # (B,1,H,W,D)
        masked = image * (1 - mask)
        out = self.unet(masked)  # (B,C,H',W',D',bins)
        cell_mask = F.interpolate(mask, size=hog.shape[2:5], mode="nearest").unsqueeze(
            -1
        )  # (B,1,H,W,D,1)

        if self.deep_supervision:
            loss = 0
            for i in range(len(out)):
                loss += self.head_weights[i] * self.loss(out[i], hog)
        else:
            loss = self.loss(out, hog)
        hl = loss * cell_mask
        hl = hl.sum() / (
            cell_mask.sum() * hog.shape[1] * hog.shape[-1] + 1e-5
        )  # masked_loss
        vl = loss * (1 - cell_mask)
        vl = vl.sum() / (
            (1 - cell_mask).sum() * hog.shape[1] * hog.shape[-1] + 1e-5
        )  # visible loss
        l = vl * self.visible_loss_weight + hl * (1 - self.visible_loss_weight)
        self.log("training loss", l, prog_bar=True, on_step=True, on_epoch=True)
        self.log("visible loss", vl, logger=True, on_epoch=True)
        self.log("masked loss", hl, logger=True, on_epoch=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        out = (out[0] if self.deep_supervision else out)
        return {
            "loss": l,
            "out": ("summary", out.detach().cpu()),
            "target": ("summary", hog.detach().cpu()),
            "diff": ("summary", (hog - out).detach().cpu())
        }

    def test_step(self, batch, _):
        image = batch[image_key]
        hog = self.hog(image)  # (B,C,H',W',D',bins)
        mask = self.mask_fn(image)  # (B,1,H,W,D)
        masked = image * (1 - mask)
        out = self.unet(masked)  # (B,C,H',W',D',bins)
        cell_mask = F.interpolate(mask, size=hog.shape[2:5], mode="nearest").unsqueeze(
            -1
        )  # (B,1,H,W,D,1)
        loss = self.loss(out, hog)
        hl = loss * cell_mask
        hl = hl.sum() / (
            cell_mask.sum() * hog.shape[1] * hog.shape[-1] + 1e-5
        )  # masked_loss
        vl = loss * (1 - cell_mask)
        vl = vl.sum() / (
            (1 - cell_mask).sum() * hog.shape[1] * hog.shape[-1] + 1e-5
        )  # visible loss
        l = vl * self.visible_loss_weight + hl * (1 - self.visible_loss_weight)
        return {
            "loss": l,
            "mask loss": hl,
            "visible loss": vl,
            "out": ("summary", out.detach().cpu()),
            "target": ("summary", hog.detach().cpu()),
            "diff": ("summary", (hog - out).detach().cpu())
        }
