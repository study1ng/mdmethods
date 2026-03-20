from experiments.mim.preprocess import PlannedSSLPreprocessor as Preprocessor
from experiments.mim.datamodule import SSLDataModule as DataModule
from experiments.mim.model import MIMModule as Model
from experiments.trainer import PlannedExperiment
from experiments.nets.u_bimamba import UBiMamba as UNet
from experiments.prune import SpacingShapeStrictPruner as Pruner
from experiments.analyze import CTAnalyzer as Analyzer
import torch


def prune(args, meta):
    Pruner(args, meta)()


def analyze(args, meta):
    Analyzer(args, meta)()


def preprocess(args, meta):
    Preprocessor(args, meta)()


class MIM(PlannedExperiment):
    def __init__(self, args, parsed):
        super().__init__(args, parsed)
        torch.set_float32_matmul_precision("medium")

    def _build_data_module(self):
        return DataModule(self.data, self.plan)

    def _build_module(self):
        unet = UNet.from_plan(
            self.plan, patch_channel=1, output_channel=1, deep_supervision=True
        )
        lm = Model(unet, mask_ratio=0.6)
        return lm


def train(args, meta):
    MIM(args, meta)()
