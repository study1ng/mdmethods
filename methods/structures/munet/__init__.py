from ..nets.UBiMambaEnc_3d import UNet
import lightning as L
from mamba_ssm import Mamba
from torch import nn
from einops import rearrange, pack, repeat
from torch.utils.checkpoint import checkpoint
import math
from monai.data.utils import iter_patch_position
import torch
from positional_encodings.torch_encodings import get_emb
import numpy as np
from monai.losses import DiceCELoss
from typing import Generator
from torch.autograd.graph import saved_tensors_hooks

class OffloadToCPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.original_device = x.device
        return x.cpu()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.to(ctx.original_device)

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
        self.div_term = np.exp(
            np.arange(0, self.channels, 2) * -(np.log(10000.0) / self.channels)
        )
        self.div_term = torch.tensor(self.div_term)
        self.register_buffer("div_term", self.div_term)

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
        self.inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2).float() / channels)
        )
        self.cached_penc = None
        self.register_buffer("inv_freq", self.inv_freq)

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


class MUNet(L.LightningModule):
    def __init__(
        self,
        unet: UNet,
        gpu_memory_limit: int,
        global_positional_encoding_proposition: float = 0.5,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.gpu_memory_limit = gpu_memory_limit  # Bytes
        self.n_stages = unet.n_stages
        # self.automatic_optimization = False
        self.patch_size = unet.input_size
        self.patch_channel = unet.input_channels
        self.skip_channels = unet.output_channels
        self.skip_size = unet.output_size
        self.encoder = unet.encoder
        self.decoder = unet.decoder
        self.bottleneck_channel = unet.output_channels[-1]
        self.bottleneck_size = unet.output_size[-1]
        self.loss = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            batch=True,
            lambda_dice=1.0,
            lambda_ce=1.0,
        )
        self.overlap_scale = 0.5  # proposion of self.patch_size
        self.overlap = tuple(int(self.overlap_scale * p) for p in self.patch_size)

        self.mamba = Mamba(
            self.bottleneck_channel, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.mamba_flip = Mamba(
            self.bottleneck_channel, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.head = nn.Linear(self.bottleneck_channel * 2, self.bottleneck_channel)
        self.layer_elements_count = tuple(
            self.skip_channels[i] * math.prod(self.skip_size[i])
            for i in range(self.n_stages)
        )
        self.global_positional_encoding_proposition = (
            global_positional_encoding_proposition
        )
        global_pe_channel = int(
            self.bottleneck_channel * self.global_positional_encoding_proposition
        )
        self.ppe = PointPositionalEncoding3D(global_pe_channel)
        self.pe = PositionalEncoding3D(self.bottleneck_channel - global_pe_channel)
        self.pos_blend = lambda a, b: a + b  # 単純に加算する

    @property
    def itemsize(self) -> int:
        if isinstance(self.dtype, torch.dtype):
            return self.dtype.itemsize

        if isinstance(self.dtype, str):
            dtype_str = self.dtype.lower()
            if "64" in dtype_str:
                return 8
            elif "32" in dtype_str:
                return 4
            elif "16" in dtype_str:
                return 2
            elif "8" in dtype_str:
                return 1

        raise ValueError(f"unable to get itemsize from {self.dtype}")

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

    def maximum_cpu_level(self, patch_len, batch_size) -> int:
        # self.type_size: self.dtypeの占用バイト数
        layer_bytesize = [
            patch_len * batch_size * lec * self.itemsize
            for lec in self.layer_elements_count
        ]
        acc = 0
        for idx, b in enumerate(layer_bytesize[::-1]):
            acc += b
            if acc > self.gpu_memory_limit:
                assert (
                    idx != 0
                ), f"""unable to place bottleneck feature map at gpu.
                memory bound: {self.gpu_memory_limit}, type: {self.dtype}, type_size: {self.itemsize},
                layer_elements_count: {self.layer_elements_count}, patch_length: {patch_len}, batch_size: {batch_size},
                bytesize_per_patch: {layer_bytesize}, 
                due to layer_bytesize[-1]={layer_bytesize[-1]} > self.gpu_memory_limit, we can conclude that it's unable."""
                return self.n_stages - idx
        return 0  # すべてgpuに載せることができる

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
            image.shape[2:], self.patch_size, overlap=self.overlap
        )
        patch_slice_iter = (
            tuple(
                slice(start, start + patch_len)
                for start, patch_len in zip(patch_start, self.patch_size)
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
        ppe_re = repeat(ppe, "c -> b l c", b=b, l=h * w * d)
        pe = self.pe.forward(patch_shape)
        pe_re = rearrange(pe, "b h w d c -> b (h w d) c")
        ret, _ = pack([ppe_re, pe_re], "b l *")
        assert ret.shape == (b, h * w * d, c)
        return ret

    def to_skips(
        self, on_cpus, on_gpus, reshaped, patches
    ) -> Generator[
        tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, tuple[int, ...]]
    ]:
        """
        Packs several generators into one.

        Returns:
            Generator[(skip feature maps, patch image, patch label, patch position)]
        """
        # on_cpus: {patch_pos: shallower feature maps}
        # on_gpus: {patch_pos: deeper feature maps}
        # reshaped: bottleneck feature map (P, B, C, H, W, D)
        # patches: list of (concated feature map, patch_image, patch_label, patch_position)
        assert (
            len(on_cpus) == len(on_gpus)
            and len(on_gpus) == len(reshaped)
            and len(reshaped) == len(patches)
        ), "Lengths of inputs must match."

        for i, (patch_image, patch_label, patch_position) in enumerate(patches):
            cpu_element = on_cpus[patch_position]
            gpu_element = on_gpus[patch_position]
            last = reshaped[i]
            skips = [*cpu_element, *gpu_element, last]
            yield (skips, patch_image, patch_label, patch_position)

    def transfer_batch_to_device(self, batch, _, __):
        return batch

    def training_step(self, batch, _):
        image, label = batch["image"], batch["label"]
        # image: the whole image which is on cpu
        # label: the whole ground truth which is on cpu
        # B, C, H, W, D, P: ボトルネックにおけるバッチサイズ, チャネル数, 高さ, 幅, 深さ, パッチ数
        on_cpus = {}
        on_gpus = {}
        lasts = []
        blend = []
        patches = self.split_to_patch(image, label)

        def to_cpu(tensor: torch.Tensor):
            return tensor.cpu()
        def to_gpu(tensor: torch.Tensor):
            return tensor.to(self.device)

        for patch_img, _, patch_pos in patches:
            patch_img = patch_img.to(self.device).requires_grad_()
            skips = checkpoint(
                self.encoder, patch_img, use_reentrant=False
            )  # avoid backward gradient OOM
            maximum_level = self.maximum_cpu_level(len(patches), patch_img.shape[0])
            on_cpu = [OffloadToCPU.apply(i) for i in skips[:maximum_level]]
            on_gpu = [i for i in skips[maximum_level:-1]]
            on_cpus[patch_pos] = on_cpu
            on_gpus[patch_pos] = on_gpu

            last = skips[-1]
            flat_transpose = rearrange(last, "b c h w d -> b (h w d) c")  # (B, HWD, C)
            pos_encode = self.add_pos_enc(last.shape, patch_pos)  # (B, HWD, C)
            blended = self.pos_blend(
                pos_encode, flat_transpose
            )  # (B, HWD, C), (B, HWD, C) -> (B, HWD, C)
            lasts.append(flat_transpose)
            blend.append(blended)
            del skips
        lasts, _ = pack(lasts, "b * c")  # (B, HWDP, C)
        blended, _ = pack(blend, "b * c")  # (B, HWDP, C)
        blended_c = blended.contiguous()  # (B, HWDP, C)
        blended_flipped = torch.flip(blended_c, dim=[1]).contiguous()  # (B, HWDP, C)

        mamba = self.mamba(blended_c)  # (B, HWDP, C)
        mamba_flipped = torch.flip(
            self.mamba_flip(blended_flipped), dim=[1]
        ).contiguous()  # (B, HWDP, C)
        mamba_cat, _ = pack([mamba, mamba_flipped], "b l *")  # (B, HWDP, 2C)
        bottleneck = self.head(mamba_cat)  # (B, HWDP, C)
        bottleneck += lasts  # 残差接続: (B, HWDP, C)
        h, w, d = self.bottleneck_size
        reshaped = rearrange(
            bottleneck,
            "b (p h w d) c -> p b c h w d",
            p=len(patches),
            h=h,
            w=w,
            d=d,  # (P, B, C, H, W, D)
        )
        all_loss = 0
        for skip, _, patch_label, __ in self.to_skips(
            on_cpus, on_gpus, reshaped, patches
        ):
            on_gpu_skip = [to_gpu(s) if s.device.type == "cpu" else s for s in skip]
            out = checkpoint(
                self.decoder, on_gpu_skip, use_reentrant=False
            )  # avoid backward gradient OOM
            patch_label = patch_label.to(self.device)
            loss = self.loss(out, patch_label)
            all_loss += loss
        all_loss /= len(patches)

        self.log("step_loss", all_loss, prog_bar=True, on_step=True)
        return all_loss
