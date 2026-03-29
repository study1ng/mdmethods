from experiments.pretrained_seg import prune, preprocess, analyze, PlainSegmentation
from experiments.nets import UBiMamba as UNet
from experiments.pretrained_seg.model import SegmentationModule as Model

class UNet1kEpochs(PlainSegmentation):
    def _build_module(self):
        unet = UNet.from_plan(self.plan, 1, deep_supervision=True, output_channel=118)
        lm = Model(unet=unet, plan=self.plan)
        return lm

def train(args, parsed):
    UNet1kEpochs(args, parsed)()

inference = train