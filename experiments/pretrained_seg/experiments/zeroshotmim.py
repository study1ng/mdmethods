from experiments.pretrained_seg import prune, analyze, preprocess, PlainSegmentation
from experiments.nets.ubimamba import UBiMamba as UNet
from experiments.pretrained_seg.model import SegmentationModule as Model

class ZeroshotMIM(PlainSegmentation):
    def _build_module(self):
        unet = UNet.from_plan(self.plan, 1, deep_supervision=True, output_channel=118)
        lm = Model(unet=unet, pretrained_path=self.args.pretrained_path, plan=self.plan)
        return lm
    
def train(args, parsed):
    ZeroshotMIM(args, parsed)()