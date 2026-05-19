from experiments.plan import Plan
from experiments.trainer import UNetTrainingModule
import torch
import experiments.config
from experiments.config import image_key, label_key
import torch.nn.functional as F
from torch import Tensor, nn
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss

from experiments.utils.wraputils import assert_to_integer


class SegmentationModule(UNetTrainingModule):
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
        else:
            loss = self.loss(out, label)
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
        self.unet.deep_supervision = False
        self.unet.decoder.deep_supervision = False
        out = self(image)
        self.unet.deep_supervision = self.deep_supervision
        self.unet.decoder.deep_supervision = self.deep_supervision
        loss = self.loss(out, label)
        self.log("validation loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return {
            "loss": loss,
            "image": (("image", "summary"), image.detach().cpu()),
            "gt": ("image", label.detach().cpu()),
            "out": ("label", out.detach().cpu()),
        }

    def test_step(self, batch, _):
        image = batch[image_key]  # (B,C,H,W,D)
        out = sliding_window_inference(
            image,
            roi_size=self.plan.patch_size,
            sw_batch_size=len(batch),
            predictor=self,
            overlap=0.5,
            mode="gaussian",
            progress=None,
            device="cpu",
            padding_mode="replicate",
        )
        return {
            "out": ("label", out.detach().cpu()),
        }
