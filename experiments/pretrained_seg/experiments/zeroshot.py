from experiments.config import default_training_config
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
        config["max_epochs"] = 100
        config["min_epochs"] = 100
        config["callbacks"].append(ZeroshotFinetuning())
        return config

def train(args, parsed):
    Zeroshot(args, parsed)()
