from experiments.plan import Plan
from experiments.trainer import UNetTrainingModule, load_from_pretrained
from pathlib import Path
import torch
import experiments.config
from experiments.config import image_key, label_key
from pprint import pprint
import torch.nn.functional as F
from torch import nn
from peft import LoraConfig, get_peft_model
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss


class SegmentationModule(UNetTrainingModule):
    def __init__(
        self,
        *,
        unet=None,
        pretrained_path: Path | str | None = None,
        weights=None,
        loss: nn.Module = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            batch=True,
            lambda_dice=1.0,
            lambda_ce=1.0,
        ),
        lora: LoraConfig | None = None,
        plan: Plan,
        **kwargs
    ):
        self.pretrained_path = pretrained_path
        if pretrained_path is not None:
            unet, original_head = load_from_pretrained(
                pretrained_path=pretrained_path, unet=unet
            )
            if isinstance(original_head, nn.ModuleList):
                original_head[0].reinitialize_unet(unet=unet, **kwargs)
            else:
                original_head.reinitialize_unet(unet=unet, **kwargs)
        unet = get_peft_model(unet, lora) if lora is not None else unet
        super().__init__(unet, weights=weights)
        self.loss = loss
        self.plan = plan

    def forward(self, x):
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
        experiments.config.assertion = self.global_step < 10
        image = batch[image_key]  # (B,C,H,W,D)
        label = batch[label_key]
        if self.global_step == 1:
            print("input shape:", image.shape)
            print("expected output shape: ")
            pprint(self.unet.calculate_output_size(image.shape))
        out = self(image)
        if self.deep_supervision:
            loss = 0
            for i in range(len(out)):
                loss += self.head_weights[i] * self.loss(
                    out[i], F.interpolate(label, out[i].shape[2:], mode="nearest")
                )
        else:
            loss = self.loss(out, label)
        self.log("training loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        return {
            "loss": loss,
            "out": (out[0] if self.deep_supervision else out).detach().cpu(),
        }

    def validation_step(self, batch, _):
        image = batch[image_key]  # (B,C,H,W,D)
        label = batch[label_key]
        print(torch.max(label))
        self.unet.deep_supervision = False
        self.unet.decoder.deep_supervision = False
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
        self.unet.deep_supervision = self.deep_supervision
        self.unet.decoder.deep_supervision = self.deep_supervision
        loss = self.loss(out, label.to("cpu")).to(self.device)
        self.log("validation loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return {
            "loss": loss,
            "out": out,
        }
