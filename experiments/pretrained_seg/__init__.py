from experiments.nets.builder import Builder
from experiments.trainer import PlannedExperiment, PlannedInferencer
from experiments.prune import NoPruner as Pruner
from experiments.analyze import CTAnalyzer as Analyzer
from experiments.pretrained_seg.datamodule import CropSegDataModule as DataModule
from experiments.pretrained_seg.model import SegmentationModule as Model
from experiments.munet.preprocess import PlannedPreprocessor as Preprocessor
import torch
from experiments.utils.fsutils import resolved_path


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

    def get_argument_parser(self):
        parser = super().get_argument_parser()
        parser.add_argument("pretrained_path", type=resolved_path, default=None)
        return parser

    def _build_data_module(self):
        return DataModule(self.data, self.plan)

    def _build_module(self):
        builder = Builder()
        if self.args.pretrained_path is not None:
            builder = builder.based_on_ckpt(self.args.pretrained_path).reinitialize("nets.plainunet.PlainHead")
        else:
            builder = builder.based_on_plan(
                self.plan,
                "nets.ubimamba.UBiMamba",
                self.plan,
                input_channel=1,
                output_channel=1,
                deep_supervision=True,
            )
        builder = builder.to_params()
        lm = Model(builder=builder, plan=self.plan)
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
        builder = Builder().based_on_ckpt(self.ckpt_path).to_params()
        lm = Model(builder, plan=self.plan)
        return lm


def inference(args, meta):
    PlainSegInferencer(args, meta)()
