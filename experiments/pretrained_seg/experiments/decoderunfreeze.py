from experiments.pretrained_seg import (
    PlainSegmentation,
    analyze,
    prune,
    inference
)
from lightning.pytorch.callbacks import BaseFinetuning


class DecoderUnfreezeFinetuning(BaseFinetuning):
    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.unet.encoder)

    def finetune_function(self, pl_module, epoch, optimizer):
        pass

class DecoderUnfreeze(PlainSegmentation):
    def configure_trainer(self, config):
        config["callbacks"].append(DecoderUnfreezeFinetuning())
        return config

def train(args, parsed):
    DecoderUnfreeze(args, parsed)()
