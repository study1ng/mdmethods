import torch.utils
import torch.utils.checkpoint

from experiments.nets.builder import Builder
from experiments.plan import Plan
from experiments.trainer import UNetTrainingModule
from monai.data.utils import iter_patch_position
import torch
from monai.losses import DiceCELoss
from typing import Generator
from experiments.config import image_key, label_key
from experiments.munet.munet_bottleneck import MUNetBottleneck, PatchFeature, PatchPosition

class MUNetTrainingModule(UNetTrainingModule):
    def __init__(
        self,
        builder: Builder,
        *,
        weights=None,
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
        super().__init__(builder, weights)
        self.save_hyperparameters()
        self.loss = loss
        self.bottleneck = MUNetBottleneck(
            self.unet.skip_channels[-1],
            global_positional_encoding_proposition,
            pos_blend,
        )

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

    def training_step(self, batch, _):
        image, label = batch[image_key], batch[label_key]
        # image: the whole image
        # label: the whole ground truth
        # B, C, H, W, D, P: ボトルネックにおけるバッチサイズ, チャネル数, 高さ, 幅, 深さ, パッチ数
        skips_map = {}
        lasts = []
        patches = self.split_to_patch(image, label)

        for patch_img, _, patch_pos in patches:
            if self.checkpoint_level <= 1:
                skips = self.unet.encoder(patch_img)
            else:
                skips = torch.utils.checkpoint.checkpoint(
                    self.unet.encoder, patch_img, use_reentrant=False
                )
            skips_map[patch_pos] = skips[:-1]
            last = skips[-1]
            lasts.append(PatchFeature(last, patch_pos))
        bottleneck_features = self.bottleneck(tuple(lasts))
        all_loss = 0
        for skip, _, patch_label, __ in self.to_skips(skips_map, bottleneck_features, patches):
            if self.checkpoint_level <= 1:
                out = self.unet.decoder(skip)
            else:
                out = torch.utils.checkpoint.checkpoint(
                    self.unet.decoder, skip, use_reentrant=False
                )
            loss = self.loss(out, patch_label)
            all_loss += loss
        all_loss /= len(patches)

        self.log("step_loss", all_loss, prog_bar=True, on_step=True)
        return all_loss

    def to_skips(
        self,
        skips_map: dict[PatchPosition, tuple[torch.Tensor, ...]],
        reshaped: tuple[PatchFeature, ...],
        patches: tuple[
            tuple[torch.Tensor, torch.Tensor, PatchPosition],
            ...
        ],
    ) -> Generator[
        tuple[
            list[torch.Tensor],
            torch.Tensor,
            torch.Tensor,
            PatchPosition,
        ],
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

        reshaped = {r.pos: r.feature for r in reshaped}

        for patch_image, patch_label, patch_position in patches:
            last = reshaped[patch_position]
            skips = [*skips_map[patch_position], last]
            yield (skips, patch_image, patch_label, patch_position)
