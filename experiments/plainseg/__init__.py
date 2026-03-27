from experiments.trainer import PlannedExperiment
from experiments.nets.ubimamba import UBiMamba as UNet
from experiments.prune import SpacingShapeStrictPruner as Pruner
from experiments.analyze import CTAnalyzer as Analyzer
from experiments.plainseg.datamodule import CropSegDataModule as DataModule
from experiments.plainseg.model import SegmentationModule as Model
from experiments.munet.preprocess import PlannedPreprocessor as Preprocessor
import torch


def prune(args, meta):
    Pruner(args, meta)()


def analyze(args, meta):
    Analyzer(args, meta)()


def preprocess(args, meta):
    Preprocessor(args, meta)()


class PlainSegmentation(PlannedExperiment):
    def __init__(self, args, parsed):
        super().__init__(args, parsed)
        torch.set_float32_matmul_precision("medium")

    def _build_data_module(self):
        return DataModule(self.data, self.plan)

    def _build_module(self):
        unet = UNet.from_plan(
            self.plan, input_channel=1, output_channel=1, deep_supervision=True
        )
        lm = Model(unet, mask_ratio=0.6)
        return lm
    
    def __call__(self):
        return super().__call__()



def train(args, meta):
    PlainSegmentation(args, meta)()
