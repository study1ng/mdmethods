from experiments.nets.builder import Builder
from experiments.plan import Plan
from experiments.trainer import UNetTrainingModule
from monai.data.utils import iter_patch_position
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss
from typing import Any, Generator
from experiments.config import image_key, label_key
from experiments.munet.munet_bottleneck import (
    MUNetBottleneck,
    PatchFeature,
    PatchPosition,
)
from monai.metrics import DiceMetric
from experiments.munet.stitch_utils import split_to_patch, stitch_logits, BlendMode


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
        overlap_scale: float = 0.25,
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
            the function to mix positional encoding and feature map, by default lambda
        """
        self.plan = plan
        self.overlap_scale = overlap_scale  # proposion of self.plan.patch_size
        self.overlap = tuple(int(self.overlap_scale * p) for p in self.plan.patch_size)
        self.global_positional_encoding_proposition = (
            global_positional_encoding_proposition
        )
        super().__init__(builder, weights)
        self.unet.deep_supervision = False
        self.unet.decoder.deep_supervision = False
        self.deep_supervision = False
        self.save_hyperparameters()
        self.loss = loss
        self.metric = DiceMetric(
            include_background=False,
            reduction="mean",
            num_classes=self.unet.output_channel,
        )
        self.bottleneck = MUNetBottleneck(
            self.unet.skip_channels[-1],
            global_positional_encoding_proposition,
            pos_blend,
        )
        self.automatic_optimization = False
        self.cache_skip_level = 1

    def split_to_patch(
        self, image: torch.Tensor, label: torch.Tensor | None = None
    ) -> tuple[tuple[torch.Tensor, torch.Tensor | None, tuple[int, ...]], ...]:
        """
        Returns:
            ((patch_image, patch_label, patch_position)*)
        """
        # -> [(patch_image, patch_label, patch_position)*]
        # patch_positionはパッチの開始位置(バッチ次元, チャネル次元は含まない)
        image_split = split_to_patch(
            image, patch_size=self.plan.patch_size, overlap=self.overlap
        )
        if label is not None:
            assert image.shape == label.shape
            label_split = split_to_patch(
                label, patch_size=self.plan.patch_size, overlap=self.overlap
            )
            for i, l in zip(image_split, label_split):
                assert l[1] == i[1]
            patches = tuple(
                (image[0], label[0], image[1])
                for image, label in zip(image_split, label_split)
            )
        else:
            patches = tuple((image[0], None, image[1]) for image in image_split)
        return patches

    def _slice_encoder(self, x: torch.Tensor, end: int) -> tuple[torch.Tensor, ...]:
        if end == 0:
            return ()
        hi = self.unet.encoder.stem(x)  # (B, C, H, W, D)
        ret = [hi]
        for stage in self.unet.encoder.stages:
            if len(ret) >= end:
                return tuple(ret)
            hi = stage(hi)
            ret.append(hi)
        return tuple(ret)


    def forward(
        self,
        patches: tuple[tuple[torch.Tensor, torch.Tensor | None, tuple[int, ...]], ...],
    ) -> Generator[Any, None, None]:
        skips_map = {}
        lasts = []

        for patch_img, _, patch_pos in patches:
            skips = self.unet.encoder(patch_img)
            skips_map[patch_pos] = skips[self.cache_skip_level:-1]
            lasts.append(PatchFeature(skips[-1], patch_pos))

        bottleneck_features = self.bottleneck(tuple(lasts))
        # ma = 0.
        # for b, l in zip(bottleneck_features, lasts):
        #     ma = max(ma, (b.feature - l.feature).abs().max().item())
        # print("bottleneck change max: ", ma)

        for skip, patch_img, patch_label, patch_pos in self.to_skips(
            skips_map, bottleneck_features, patches
        ):
            skips = self._slice_encoder(patch_img, self.cache_skip_level)
            bottleneck_skips = (*skips, *skip)

            out = self.unet.decoder(bottleneck_skips)
            # p_out = self.unet.decoder(skips)
            # print("last max between two networks", last.abs().max().item())
            # print("last mean between two networks", last.abs().mean().item())
            # print("diff max between two networks", (out - p_out).abs().max().item())
            # print("diff mean between two networks", (out - p_out).abs().mean().item())
            yield patch_img, patch_label, out, patch_pos

    def training_step(self, batch, _):
        image, label = batch[image_key], batch[label_key]
        # image: the whole image
        # label: the whole ground truth
        # B, C, H, W, D, P: ボトルネックにおけるバッチサイズ, チャネル数, 高さ, 幅, 深さ, パッチ数
        patches = self.split_to_patch(image, label)
        try:
            outs = self(patches)
            all_loss = 0.0
            results = []
            opt = self.optimizers()
            opt.zero_grad()
            for patch_img, patch_label, out, patch_pos in outs:
                # p_out = self.unet(patch_img)
                # if self.deep_supervision:
                #     loss = 0
                #     for i in range(len(p_out)):
                #         loss += self.head_weights[i] * self.loss(
                #             p_out[i], F.interpolate(patch_label, p_out[i].shape[2:], mode="nearest")
                #         )
                #     top_loss = self.loss(p_out[0], label)
                # else:
                #     loss = self.loss(p_out, patch_label)
                #     top_loss = loss
                loss = self.loss(out, patch_label) / len(patches)
                # print(top_loss, loss * len(patches))
                # print(self.global_step)
                self.manual_backward(loss, retain_graph=True)
                all_loss += loss.detach()
                results.append((out.detach(), patch_pos))
            scheduler = self.lr_schedulers()
            opt.step()
            scheduler.step()
        except Exception as e:
            print(image.shape)
            raise e
        # all_loss /= len(patches)

        self.log(
            "gamma",
            self.bottleneck.gamma.abs().max().item(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log("training loss", all_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        with torch.no_grad():
            out = stitch_logits(
                tuple(results),
                BlendMode,
                output_size=image.shape[2:],
            )
        pred = out.argmax(1, keepdim=True)
        return {
            "loss": all_loss,
            "image": ("image", image.detach().cpu()),
            "gt": ("image", label.detach().cpu()),
            "out": ("image", pred.detach().cpu()),
        }

    def to_skips(
        self,
        skips_map: dict[PatchPosition, tuple[torch.Tensor, ...]],
        reshaped: tuple[PatchFeature, ...],
        patches: tuple[tuple[torch.Tensor, torch.Tensor | None, PatchPosition], ...],
    ) -> Generator[
        tuple[
            tuple[torch.Tensor, ...],
            torch.Tensor,
            torch.Tensor | None,
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
        assert len(reshaped) == len(patches), "Lengths of inputs must match."

        reshaped = {r.pos: r.feature for r in reshaped}

        for patch_image, patch_label, patch_position in patches:
            last = reshaped[patch_position]            
            skips = (*skips_map[patch_position], last)
            yield (skips, patch_image, patch_label, patch_position)

    def validation_step(self, batch, _):
        image = batch[image_key]  # (B,C,H,W,D)
        label = batch[label_key]
        patches = self.split_to_patch(image, label)
        outs = self(patches)
        out = stitch_logits(
            tuple((out, patch_pos) for _, __, out, patch_pos in outs),
            BlendMode,
            output_size=image.shape[2:],
        )
        pred = out.argmax(1, keepdim=True)
        self.metric(pred, label)
        return {
            "image": ("image", image.detach().cpu()),
            "gt": ("image", label.detach().cpu()),
            "out": ("image", pred.detach().cpu()),
        }

    def on_validation_epoch_end(self):
        dice = self.metric.aggregate()
        self.log(
            "val_dice",
            dice,
            prog_bar=True,
            sync_dist=True,
        )
        self.metric.reset()

    def test_step(self, batch, _):
        image = batch[image_key]  # (B,C,H,W,D)
        patches = self.split_to_patch(image)
        outs = self(patches)
        out = stitch_logits(
            tuple((out, patch_pos) for _, __, out, patch_pos in outs),
            BlendMode,
            output_size=image.shape[2:],
        )
        pred = out.argmax(1, keepdim=True)
        return {
            "image": ("image", image.detach().cpu()),
            "out": ("image", pred.detach().cpu()),
        }

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            [
                {
                    "params": self.unet.parameters(),
                    "lr": 4e-4,
                },
                {
                    "params": self.bottleneck.parameters(),
                    "lr": 4e-4,
                },
            ],
            eps=1e-5,
            weight_decay=1e-1,
            betas=(0.9, 0.95),
        )
        total_steps = self.trainer.estimated_stepping_batches
        endless = total_steps == float("inf")
        warmup_steps = min(total_steps, 250 * 1000) // 10
        scheduler = (
            torch.optim.lr_scheduler.SequentialLR(
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
            if not endless
            else torch.optim.lr_scheduler.SequentialLR(
                optim,
                [
                    torch.optim.lr_scheduler.LinearLR(
                        optim,
                        start_factor=1e-10,
                        end_factor=1.0,
                        total_iters=warmup_steps,
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optim,
                        T_0=100000,
                        T_mult=2,
                        eta_min=1e-5,
                    ),
                ],
                milestones=[warmup_steps],
            )
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
