from dataclasses import dataclass
from typing import Generator

from experiments.nets.ubimamba import BiMambaBlock
import torch
import numpy as np
import lightning as L
from einops import rearrange, pack

PatchPosition = tuple[int, ...]


@dataclass
class PatchFeature:
    feature: torch.Tensor
    pos: PatchPosition


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PointPositionalEncoding3D(L.LightningModule):
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


class PositionalEncoding3D(L.LightningModule):
    def __init__(self, channels):
        super().__init__()

        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1

        self.channels = channels

        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))

        self.register_buffer("inv_freq", inv_freq)

        self.cached_penc = None
        self.cached_shape = None

    def forward(self, shape: tuple[int, int, int, int, int]) -> torch.Tensor:
        device = self.inv_freq.device
        dtype = self.inv_freq.dtype

        if (
            self.cached_penc is not None
            and self.cached_shape == shape
            and self.cached_penc.device == device
            and self.cached_penc.dtype == dtype
        ):
            return self.cached_penc

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
        self.cached_shape = shape

        return self.cached_penc


class MUNetBottleneck(L.LightningModule):
    def __init__(
        self,
        channel: int,
        global_positional_encoding_proposition: float = 0.5,
        pos_blend=lambda a, b: a + b,
    ):
        self.channel = channel
        self.global_positional_encoding_proposition = (
            global_positional_encoding_proposition
        )
        self.pos_blend = pos_blend
        super().__init__()
        self.bottleneck = BiMambaBlock(channel, channel)
        global_pe_channel = int(global_positional_encoding_proposition * channel)
        self.ppe = PointPositionalEncoding3D(global_pe_channel)
        self.pe = PositionalEncoding3D(channel - global_pe_channel)

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

    def forward(self, lasts: tuple[PatchFeature, ...]) -> tuple[PatchFeature, ...]:
        flats = []
        blended_list = []

        for patch in lasts:
            h, w, d = patch.feature.shape[2:]

            flat = rearrange(
                patch.feature,
                "b c h w d -> b (h w d) c",
            )  # (B, HWD, C)

            pos_encode = self.add_pos_enc(
                patch.feature.shape,
                patch.pos,
            ).to(dtype=self.dtype)

            blended = self.pos_blend(
                pos_encode,
                flat,
            )

            flats.append(flat)
            blended_list.append(blended)

        residual, _ = pack(flats, "b * c")  # (B, HWDP, C)
        blended, _ = pack(blended_list, "b * c")  # (B, HWDP, C)

        residual = rearrange(
            residual,
            "b (p h w d) c -> b c p h w d",
            p=len(lasts),
            h=h,
            w=w,
            d=d,
        )
        blended = rearrange(
            blended,
            "b (p h w d) c -> b c p h w d",
            p=len(lasts),
            h=h,
            w=w,
            d=d,
        )

        bottleneck = self.bottleneck(blended.contiguous()) + residual

        reshaped = rearrange(
            bottleneck,
            "b c p h w d -> p b c h w d",
            p=len(lasts),
            h=h,
            w=w,
            d=d,
        )

        return tuple(
            PatchFeature(
                feature=reshaped[i],
                pos=lasts[i].pos,
            )
            for i in range(len(lasts))
        )
