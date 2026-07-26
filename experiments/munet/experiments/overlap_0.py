from experiments.munet import BottleneckSeg
from experiments.nets.builder import Builder
from experiments.pretrained_seg import PlainSegmentation, analyze, prune, inference
from lightning.pytorch.callbacks import BaseFinetuning
from experiments.munet.datamodule import NoCropDataModule as DataModule
from experiments.munet.model import MUNetTrainingModule as Model


class Overlap0(BottleneckSeg):
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
    Overlap0(args, parsed)()
