from experiments.pretrained_seg import (
    PlainSegmentation,
    analyze,
    prune,
    inference
)
from lightning.pytorch.callbacks import BaseFinetuning


class BottleneckFinetuning(BaseFinetuning):
    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.unet)

    def finetune_function(self, pl_module, epoch, optimizer):
        pass
    
class BottleneckSeg(PlainSegmentation):
    def configure_trainer(self, config):
        config["callbacks"].append(BottleneckFinetuning())
        return config

def train(args, parsed):
    BottleneckSeg(args, parsed)()
