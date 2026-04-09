from experiments.nets.builder import Builder
from experiments.trainer import PlannedExperiment
from experiments.prune import SpacingShapeStrictPruner as Pruner
from experiments.analyze import CTAnalyzer as Analyzer
from experiments.mim.datamodule import SSLDataModule as DataModule
from experiments.maskfeat.model import MaskFeatModule as Model
from experiments.mim.preprocess import PlannedSSLPreprocessor as Preprocessor
import torch


def prune(args, meta):
    Pruner(args, meta)()


def analyze(args, meta):
    Analyzer(args, meta)()


def preprocess(args, meta):
    Preprocessor(args, meta)()


class MaskFeat(PlannedExperiment):
    def __init__(self, args, parsed):
        super().__init__(args, parsed)
        torch.set_float32_matmul_precision("medium")

    def _build_data_module(self):
        return DataModule(self.data, self.plan)

    def _build_module(self):
        builder = Builder().based_on_plan(
            "nets.ubimamba.UBiMamba",
            self.plan,
            input_channel=1,
            output_channel=1,
            deep_supervision=True,
        ).to_params()
        lm = Model(builder, mask_ratio=0.6)
        return lm


def train(args, meta):
    MaskFeat(args, meta)()
