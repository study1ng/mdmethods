from experiments.nets.plainunet import PlainUNet
from experiments.plan import Plan
from experiments.trainer import UNetTrainingModule
import torch
import experiments.config
from experiments.config import image_key, label_key
import torch.nn.functional as F
from torch import Tensor, nn
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric


class SegmentationModule(UNetTrainingModule):
    unet: PlainUNet
    def __init__(
        self,
        builder: list[dict] = None,
        *,
        weights=None,
        loss: nn.Module = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            batch=True,
            lambda_dice=1.0,
            lambda_ce=1.0,
        ),
        plan: Plan,
    ):
        super().save_hyperparameters()
        super().__init__(builder, weights=weights)
        self.loss = loss
        self.metric = DiceMetric(include_background=False, reduction="mean", num_classes=self.unet.output_channel)
        self.plan = plan

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, _):
        experiments.config.assertion = self.global_step < 10
        image = batch[image_key]  # (B,C,H,W,D)
        label = batch[label_key]
        out = self(image)
        if self.deep_supervision:
            loss = 0
            for i in range(len(out)):
                loss += self.head_weights[i] * self.loss(
                    out[i], F.interpolate(label, out[i].shape[2:], mode="nearest")
                )
            top_loss = self.loss(out[0], label)
        else:
            loss = self.loss(out, label)
            top_loss = loss
        if loss < 0.0:
            print(loss)
            print(self.head_weights)
            print(self.loss, self.loss.lambda_dice, self.loss.lambda_ce)
            print("label: ", label.shape)
            if self.deep_supervision:
                for i in range(len(out)):
                    print(out[i].shape)
            else:
                print(out.shape)
            raise AssertionError("loss < 0")
        self.log("training loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "training top loss",
            top_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        out = out[0] if self.deep_supervision else out
        return {
            "loss": loss,
            "image": (("image", "summary"), image.detach().cpu()),
            "gt": ("image", label.detach().cpu()),
            "out": ("label", out.detach().cpu()),
        }

    def validation_step(self, batch, _):
        image = batch[image_key]  # (B,C,H,W,D)
        label = batch[label_key]
        out = sliding_window_inference(
            image,
            roi_size=self.plan.patch_size,
            sw_batch_size=1,
            predictor=self,
            overlap=0.5,
            mode="gaussian",
            progress=None,
            device=self.device,
            padding_mode="replicate",
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
        out = sliding_window_inference(
            image,
            roi_size=self.plan.patch_size,
            sw_batch_size=1,
            predictor=self,
            overlap=0.5,
            mode="gaussian",
            progress=None,
            device=self.device,
            padding_mode="replicate",
        )
        return {
            "out": ("label", out.detach().cpu()),
        }


    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            [
                {
                    "params": self.unet.decoder.head.parameters(),
                    "lr": 4e-4,
                    "weight_decay": 1e-1,
                },
                {
                    "params": self.unet.encoder.parameters(),
                    "lr": 4e-5,
                    "weight_decay": 5e-2,
                },
                {
                    "params": self.unet.decoder.stages.parameters(),
                    "lr": 4e-5,
                    "weight_decay": 5e-2,
                }
            ],
            eps=1e-5,
            betas=(0.9, 0.95),
        )
        total_steps = self.trainer.estimated_stepping_batches
        endless = total_steps == float("inf")
        warmup_steps = min(total_steps, 250 * 1000) // 100
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
                    optim, T_max=total_steps - warmup_steps, eta_min=1e-6
                ),
            ],
            milestones=[warmup_steps],
        ) if not endless else torch.optim.lr_scheduler.SequentialLR(
            optim, [
                torch.optim.lr_scheduler.LinearLR(
                    optim,
                    start_factor=1e-10,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                ),
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optim, T_0=100000, T_mult=2, eta_min=1e-6,
                )
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
