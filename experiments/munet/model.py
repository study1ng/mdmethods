import torch.utils
import torch.utils.checkpoint

from experiments.nets.builder import Builder
from experiments.nets.ubimamba import BiMambaBlock
from experiments.plan import Plan
from experiments.trainer import UNetTrainingModule
from torch import nn
from einops import rearrange, pack
from monai.data.utils import iter_patch_position
import torch
import numpy as np
from monai.losses import DiceCELoss
from typing import Generator
from experiments.config import image_key, label_key


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PointPositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super().__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        div_term = np.exp(
            np.arange(0, self.channels, 2) * -(np.log(10000.0) / self.channels)
        )
        div_term = torch.tensor(div_term)
        self.register_buffer("div_term", div_term)

    def forward(self, pos: tuple[int, int, int]) -> torch.Tensor:
        device = self.div_term.device
        dtype = self.div_term.dtype
        pe = torch.zeros(self.channels * 3, device=device, dtype=dtype)
        h, w, d = pos
        pe[0 : self.channels : 2] = torch.sin(h * self.div_term)
        pe[1 : self.channels : 2] = torch.cos(h * self.div_term)
        pe[self.channels : 2 * self.channels : 2] = torch.sin(w * self.div_term)
        pe[self.channels + 1 : 2 * self.channels : 2] = torch.cos(w * self.div_term)
        pe[2 * self.channels : 3 * self.channels : 2] = torch.sin(d * self.div_term)
        pe[2 * self.channels + 1 : 3 * self.channels : 2] = torch.cos(d * self.div_term)
        return pe[: self.org_channels]


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super().__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.cached_penc = None
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, shape: tuple[int, int, int, int, int]) -> torch.Tensor:
        """
        :param shape: A 5d shape tuple (b, c, h, w, d)
        :return: Positional Encoding Matrix of size (b, h, w, d, ch)
        """
        device = self.inv_freq.device
        dtype = self.inv_freq.dtype

        if (
            self.cached_penc is not None
            and self.cached_penc.shape == shape
            and self.cached_penc.device == device
        ):
            return self.cached_penc

        self.cached_penc = None
        batch_size, _, h, w, d = shape
        pos_x = torch.arange(h, device=device, dtype=dtype)
        pos_y = torch.arange(w, device=device, dtype=dtype)
        pos_z = torch.arange(d, device=device, dtype=dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros(
            (h, w, d, self.channels * 3),
            device=device,
            dtype=dtype,
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, : self.org_channels].repeat(
            batch_size, 1, 1, 1, 1
        )
        return self.cached_penc


class MUNetTrainingModule(UNetTrainingModule):
    def __init__(
        self,
        builder: Builder,
        *,
        weights = None,
        loss=DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            batch=True,
            lambda_dice=1.0,
            lambda_ce=1.0,
        ),
        global_positional_encoding_proposition: float = 0.5,
        pos_blend=lambda a, b: a + b,
        plan: Plan,
        checkpoint_level: int = 0,
    ):
        """Training Module for MUNet

        Parameters
        ----------
        builder : Builder
            Builder to build
        weights : None | float | list[float]
            the weight which would be applied if deep_supervision, 
            default to [1/2, 1/4, 1/8, ...], 
            whose sum would be adjusted to 1. 
            first element is about head
        plan : Plan
            plan
        loss : loss, optional
            loss function, by default DiceCELoss( include_background=False, to_onehot_y=True, softmax=True, batch=True, lambda_dice=1.0, lambda_ce=1.0, )
        global_positional_encoding_proposition : float, optional
            the proposition of positional encoding which represents the position the patch in whole image, by default 0.5
        pos_blend : (Tensor, Tensor) -> Tensor, optional
            the function to mix positional encoding and feature map, by default lambdaa
        checkpoint_level : int, optional
            the level of checkpointing, by default 0
            higher then slower less memory consumption
        """
        self.checkpoint_level = checkpoint_level
        self.plan = plan
        self.overlap_scale = 0.5  # proposion of self.plan.patch_size
        self.overlap = tuple(int(self.overlap_scale * p) for p in self.plan.patch_size)
        self.global_positional_encoding_proposition = (
            global_positional_encoding_proposition
        )
        global_pe_channel = int(
            global_positional_encoding_proposition * self.unet.skip_channels[-1]
        )
        super().__init__(builder, weights)
        self.save_hyperparameters()
        self.loss = loss
        self.bottleneck = BiMambaBlock(
            self.unet.skip_channels[-1], self.unet.skip_channels[-1]
        )
        self.ppe = PointPositionalEncoding3D(global_pe_channel)
        self.pe = PositionalEncoding3D(self.unet.skip_channels[-1] - global_pe_channel)
        self.pos_blend = pos_blend


    def configure_optimizers(self):
        self.optim = torch.optim.AdamW(
            self.parameters(),
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

    def split_to_patch(
        self, image: torch.Tensor, label: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, tuple[int, ...]], ...]:
        """
        Returns:
            ((patch_image, patch_label, patch_position)*)
        """
        # -> [(patch_image, patch_label, patch_position)*]
        # patch_positionはパッチの開始位置(バッチ次元, チャネル次元は含まない)
        assert image.shape == label.shape
        patch_start_gen = iter_patch_position(
            image.shape[2:], self.plan.patch_size, overlap=self.overlap
        )
        patch_slice_iter = (
            tuple(
                slice(start, start + patch_len)
                for start, patch_len in zip(patch_start, self.plan.patch_size)
            )
            for patch_start in patch_start_gen
        )
        patches = tuple(
            (
                image[(...,) + slice_tuple],
                label[(...,) + slice_tuple],
                tuple(s.start for s in slice_tuple),
            )
            for slice_tuple in patch_slice_iter
        )
        return patches

    def add_pos_enc(self, patch_shape, patch_pos) -> torch.Tensor:
        """
        Returns:
            positional encoding which shape is (B, H*W*D, C)
        """
        ppe = self.ppe.forward(patch_pos)  # (c)
        b, c, h, w, d = patch_shape
        ppe_re = ppe.view(1, 1, -1).expand(b, h * w * d, -1)
        pe = self.pe.forward(patch_shape)
        pe_re = rearrange(pe, "b h w d c -> b (h w d) c")
        ret, _ = pack([ppe_re, pe_re], "b l *")
        assert ret.shape == (b, h * w * d, c)
        return ret

    def training_step(self, batch, _):
        image, label = batch[image_key], batch[label_key]
        # image: the whole image
        # label: the whole ground truth
        # B, C, H, W, D, P: ボトルネックにおけるバッチサイズ, チャネル数, 高さ, 幅, 深さ, パッチ数
        skips_map = {}
        lasts = []
        blend = []
        patches = self.split_to_patch(image, label)

        for patch_img, _, patch_pos in patches:
            if self.checkpoint_level <= 1:
                skips = self.unet.encoder(patch_img)
            else:
                skips = torch.utils.checkpoint.checkpoint(self.unet.encoder, patch_img, use_reentrant=False)
            skips_map[patch_pos] = skips[:-1]
            last = skips[-1]
            h, w, d = last.shape[2:]
            flat_transpose = rearrange(last, "b c h w d -> b (h w d) c")  # (B, HWD, C)
            pos_encode = self.add_pos_enc(last.shape, patch_pos).to(
                dtype=self.dtype
            )  # (B, HWD, C)
            blended = self.pos_blend(
                pos_encode, flat_transpose
            )  # (B, HWD, C), (B, HWD, C) -> (B, HWD, C)
            lasts.append(flat_transpose)
            blend.append(blended)

        lasts, _ = pack(lasts, "b * c")  # (B, HWDP, C)
        blended, _ = pack(blend, "b * c")  # (B, HWDP, C)
        blended_c = blended.contiguous()  # (B, HWDP, C)
        if self.checkpoint_level <= 0:
            bottleneck = self.bottleneck(blended_c)  # (B, HWDP, C)
        else:
            bottleneck = torch.utils.checkpoint.checkpoint(self.bottleneck, blended_c, use_reentrant=False)
        bottleneck += lasts  # 残差接続: (B, HWDP, C)
        reshaped = rearrange(
            bottleneck,
            "b (p h w d) c -> p b c h w d",
            p=len(patches),
            h=h,
            w=w,
            d=d,  # (P, B, C, H, W, D)
        )
        all_loss = 0
        for skip, _, patch_label, __ in self.to_skips(skips_map, reshaped, patches):
            if self.checkpoint_level <= 1:
                out = self.decoder(skip)
            else:
                out = torch.utils.checkpoint.checkpoint(self.unet.decoder, skip, use_reentrant=False)
            loss = self.loss(out, patch_label)
            all_loss += loss
        all_loss /= len(patches)

        self.log("step_loss", all_loss, prog_bar=True, on_step=True)
        return all_loss

    def to_skips(self, skips_map, reshaped, patches) -> Generator[
        tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, tuple[int, ...]],
        None,
        None,
    ]:
        """
        Packs several generators into one.

        Returns:
            Generator[(skip feature maps, patch image, patch label, patch position)]
        """
        assert len(skips_map) == len(reshaped) and len(reshaped) == len(
            patches
        ), "Lengths of inputs must match."

        for i, (patch_image, patch_label, patch_position) in enumerate(patches):
            last = reshaped[i]
            skips = [*skips_map[patch_position], last]
            yield (skips, patch_image, patch_label, patch_position)
