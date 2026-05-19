from experiments.config import default_training_config
from experiments.nets.builder import Builder
from experiments.pretrained_seg import (
    PlainSegmentation,
    analyze,
    prune,
    inference
)
from experiments.nets.ubimamba import UBiMamba as UNet
from experiments.pretrained_seg.model import SegmentationModule as Model
from lightning.pytorch.callbacks import BaseFinetuning
from lightning import Trainer
import torch


class ZeroshotFinetuning(BaseFinetuning):
    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.unet.encoder)
        self.freeze(pl_module.unet.decoder.stages)

    def finetune_function(self, pl_module, epoch, optimizer):
        pass

class Zeroshot(PlainSegmentation):
    def configure_trainer(self, config):
        config["callbacks"].append(ZeroshotFinetuning())
        return config
    

    def _build_module(self):
        builder = Builder()
        if self.args.pretrained_path is not None:
            builder = builder.based_on_ckpt(self.args.pretrained_path).reinitialize(
                "nets.head_variants.MambaConvHead",
                output_channel = 118,
            )
        else:
            builder = builder.based_on_plan(
                self.plan,
                "nets.ubimamba.UBiMamba",
                self.plan,
                input_channel=1,
                output_channel=118,
                deep_supervision=True,
            )
        builder = builder.to_params()
        lm = Model(builder=builder, plan=self.plan)
        return lm

def train(args, parsed):
    Zeroshot(args, parsed)()
