from experiments.trainer import Inferencer, PlannedExperiment, PlannedInferencer
from experiments.prune import NoPruner as Pruner
from experiments.analyze import CTAnalyzer as Analyzer
from experiments.pretrained_seg.datamodule import CropSegDataModule as DataModule
from experiments.pretrained_seg.model import SegmentationModule as Model
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
        lm = Model(pretrained_path=self.args.pretrained_path, plan=self.plan)
        return lm

def train(args, meta):
    PlainSegmentation(args, meta)()


class PlainSegInferencer(PlannedInferencer):
    def __init__(self, args, parsed):
        super().__init__(args, parsed)
        torch.set_float32_matmul_precision("medium")

    def _build_data_module(self):
        return DataModule(self.preprocessed, self.plan)
    
    def _build_module(self):
        lm = Model(pretrained_path=self.ckpt_path, plan=self.plan)
        return lm


def inference(args, meta):
    PlainSegInferencer(args, meta)()
