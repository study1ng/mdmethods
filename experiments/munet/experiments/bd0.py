from experiments.nets.base import UNet, UNetReinitializer
from experiments.nets.builder import Builder
from experiments.pretrained_seg import PlainSegmentation, analyze, prune, inference
from lightning.pytorch.callbacks import BaseFinetuning
from experiments.munet.datamodule import NoCropDataModule as DataModule
from experiments.munet.model import MUNetTrainingModule as Model

class DecoderFinetuning(BaseFinetuning):
    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module: Model):
        self.freeze(pl_module.unet.encoder)
        for p in pl_module.unet.encoder.parameters():
            p.requires_grad_(False)

    def finetune_function(self, pl_module, epoch, optimizer):
        pass


class BottleneckSeg(PlainSegmentation):
    def configure_trainer(self, config):
        config = super().configure_trainer(config)
        config["callbacks"].append(DecoderFinetuning())
        return config
    
    def _build_data_module(self):
        return DataModule(self.data, self.plan)

    def _build_module(self):
        builder = Builder()
        if self.args.pretrained_path is not None:
            builder = builder.based_on_ckpt(self.args.pretrained_path)
        else:
            raise Exception("Munet needs pretrained model")
        builder = builder.to_params()
        lm = Model(builder=builder, plan=self.plan, overlap_scale=0., gamma=1e-5)
        return lm


def train(args, parsed):
    BottleneckSeg(args, parsed)()
