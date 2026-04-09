from experiments.nets.builder import Builder
from experiments.pretrained_seg import prune, preprocess, analyze, PlainSegmentation
from experiments.pretrained_seg.model import SegmentationModule as Model


class UNet1kEpochs(PlainSegmentation):
    def _build_module(self):
        builder = Builder.based_on_plan("nets.UBiMamba", self.plan, 1, deep_supervision=True, output_channel=118).to_params()
        lm = Model(builder, plan=self.plan)
        return lm


def train(args, parsed):
    UNet1kEpochs(args, parsed)()


inference = train
