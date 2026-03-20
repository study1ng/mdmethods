from functools import partial

import einops.layers
import einops.layers.torch
from torch import Tensor, nn, tensor
import torch
from experiments.nets.plainunet import SingleConvBlock
from experiments.utils import (
    assert_divisable,
    get_gaussian_kernel,
    assert_eq,
    prolong,
    element_wise2,
)
import math
import einops
import torch.nn.functional as F
from experiments.mim.model import generate_mask
from math import prod
from experiments.config import image_key
import lightning as L
import torch
from experiments.nets import UNet, UNetHead
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
        self.cell_size = prolong(cell_size, 3, int)
        self.block_size = prolong(block_size, 3, int)
        self.spacing = prolong(spacing, (int, float), 3)
        self.gaussian_window_size = prolong(gaussian_window_size, int, 3)
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
        m = torch.diag(1. / torch.tensor(self.spacing, dtype=torch.float))
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

        @element_wise2
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
            repeat_rate = assert_divisable(norm.shape[2:], self.gaussian_window_size)
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

        h_out, w_out, d_out = assert_divisable((h, w, d), self.cell_size)
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
        assert_eq(5, len(input_size))
        b, c, h, w, d = input_size

        @element_wise2
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


class HogHead(UNetHead):
    def __init__(
        self, input_size, input_channel, output_size, output_channel, hog_channel: int
    ):
        super().__init__(
            input_size, input_channel, output_size + (hog_channel,), output_channel
        )
        self.scales = assert_divisable(input_size, output_size)
        self.hog_channel = hog_channel
        self.ps = nn.Sequential(
            einops.layers.torch.Rearrange(
                "b c (h p0) (w p1) (d p2) -> b (c p0 p1 p2) h w d",
                p0=self.scales[0],
                p1=self.scales[1],
                p2=self.scales[2],
            ),
            SingleConvBlock(
                input_channel=input_channel * prod(self.scales),
                output_channel=output_channel * hog_channel,
                size=output_size,
                kernel_size=1,
            ),
            einops.layers.torch.Rearrange(
                "b (c p) h w d -> b c h w d p",
                p=self.hog_channel
            )
        )

    def _forward(self, x):
        return self.ps(x)

    @classmethod
    def attach_to_unet(cls, unet: UNet, hog: HogLayer3D):
        assert (
            hog.block_size == (1,1,1)
        ), "We don't support for hog block normalization cuz that will make output hog resolution not divisors of input image resolution"
        unet_output_shape = (1, unet.output_channel, *unet.patch_size)
        output_shape = hog.output_size(unet_output_shape)
        output_channel = unet.patch_channel # HOG channel would be as same as input channel
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
                        hog_channel=hog_channel,
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
    """UNet Pretraining by predict HOG, inspired by MaskFeat <https://arxiv.org/abs/2112.09133>

    Parameters
    ----------
    unet : UNet
        Instance of UNet
    mask_ratio : float
        Ratio of mask
    mask_fn : function which argument is (image, mask_size, mask_ratio, device=None), and return a binary mask which shape is same as image
        Function which generate mask
    mask_size : tuple[int, int, int] | None
        Mask size. 
    weights : float | tuple[float, ...] | None = None
        Weights used if deep supervision.
        If weights is float, the k nd stage's weight would be weights ** k.
        The weights would be normalized.
        Length of weights should be as same as unet.n_stages + 1
    loss_fn : function which argument is (image, target), and return a unreducted feature map
        Loss function
    cell_size : tuple[int, int, int] | int | None = None
        HOG cell size. if not assigned, it would be a mask size divisor which is nearest from unet bottleneck feature map size
    gaussian_window_size : tuple[int, int, int] | int | None = None
        HOG gaussian window size. If None then gaussian window would not be adopted
    signed: bool = True
        If use signed HOG or unsigned HOG
    visible_loss_weight: float = 0.0
        The loss weight for area which is not masked.

    Notes
    -----
    HOG block size is fixed to 1, meaning we will not use block normalization.
    This is intended not to break the spatial relationship between mask patch and cell.
    """
    def __init__(
        self,
        unet: UNet,
        mask_ratio: float,
        mask_fn=generate_mask,
        mask_size: tuple[int, ...] | None = None,
        weights: float | tuple[float, ...] | None = None,
        loss_fn=partial(nn.MSELoss, reduction="none"),
        cell_size: tuple[int, ...] | int | None = None,
        gaussian_window_size: tuple[int, int, int] | int | None = None,
        signed: bool = True,
        visible_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["unet"])
        self.deep_supervision = unet.deep_supervision
        self.mask_fn = mask_fn
        self.mask_ratio = mask_ratio
        self.mask_size = mask_size
        self.weights = weights
        self.visible_loss_weight = visible_loss_weight
        self.gaussian_window_size = gaussian_window_size
        self.signed = signed
        self.loss = loss_fn()

        if mask_size is None:
            self.mask_size = assert_divisable(
                self.unet.patch_size, self.unet.skip_size[-1]  # (16,16,16)
            )
            print("default mask size: ", self.mask_size)

        if cell_size is None:
            self.cell_size = self.unet.skip_size[-1]  # (7,7,8)

            @element_wise2
            def _find_closest_divisor(i, j):
                """iの約数のうち, 最もjと値が近いものを返す"""
                divisors = sympy.divisors(i)
                closest = min(divisors, key=lambda x: (abs(x - j), x))
                return closest

            # mask_sizeの約数となるように調整
            self.cell_size = _find_closest_divisor(self.mask_size, self.cell_size)
            print("default cell size: ", self.cell_size)

        self.cell_per_mask = assert_divisable(
            self.mask_size, self.cell_size
        )

        self.hog = HogLayer3D(
                    cell_size=self.cell_size,
                    block_size=1,  # ブロックでの正規化は行わない
                    gaussian_window_size=self.gaussian_window_size,
                    signed=self.signed,
                )
        self.unet = HogHead.attach_to_unet(unet, self.hog)
        print(self.unet)

        if self.deep_supervision:
            if isinstance(self.weights, tuple):
                assert_eq(len(self.unet.decoder.head), len(self.weights))
            if self.weights is None:
                self.weights = 2.0
            if isinstance(self.weights, float):
                self.weights = tuple(
                    self.weights**i for i in range(len(self.unet.decoder.head))
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
        hog = self.hog(image)  # (B,C,H',W',D',bins)
        mask = self.mask_fn(
            image, self.mask_size, self.mask_ratio, self.device
        )  # (B,1,H,W,D)
        # masked = image * (1 - mask) + self.mask_token * mask
        masked = image * (1 - mask)
        out = self.unet(masked)  # (B,C,H',W',D',bins)
        cell_mask = F.interpolate(
            mask, size=hog.shape[2:5], mode="nearest"
        ).unsqueeze(-1) # (B,1,H,W,D,1)

        if self.deep_supervision:
            loss = 0
            for i in range(len(out)):
                loss += self.head_weights[i] * self.loss(out[i], hog)
        else:
            loss = self.loss(out, hog)
        l = loss * cell_mask
        l = l.sum() / (cell_mask.sum() * hog.shape[1] * hog.shape[-1] + 1e-5)  # masked_loss
        vl = loss * (1 - cell_mask)
        vl = vl.sum() / ((1 - cell_mask).sum() * hog.shape[1] * hog.shape[-1] + 1e-5) # visible loss
        l = vl * self.visible_loss_weight + l * (1 - self.visible_loss_weight)
        self.log("training_loss", l, prog_bar=True, on_epoch=True)
        return l
