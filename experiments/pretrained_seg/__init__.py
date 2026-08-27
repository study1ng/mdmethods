from experiments.nets.builder import Builder
from experiments.trainer import PlannedExperiment, PlannedInferencer
from experiments.prune import NoPruner as Pruner
from experiments.analyze import CTAnalyzer as Analyzer
from experiments.pretrained_seg.datamodule import CropSegDataModule as DataModule
from experiments.pretrained_seg.model import SegmentationModule as Model
from experiments.argument_adaptor import ArgumentAdaptor
from experiments.plan import Plan
import torch
from experiments.utils.fsutils import resolved_path
from experiments.utils import nowstring


def prune(args, meta):
    Pruner(args, meta)()


def analyze(args, meta):
    Analyzer(args, meta)()


class PlainSegmentation(PlannedExperiment):
    def __init__(self, args, parsed):
        super().__init__(args, parsed)
        torch.set_float32_matmul_precision("medium")

    def get_argument_parser(self):
        parser = super().get_argument_parser()
        parser.add_argument("--pretrained_path", type=resolved_path, default=None)
        return parser

    def _build_data_module(self):
        return DataModule(self.data, self.plan)

    def _build_module(self):
        builder = Builder()
        if self.args.pretrained_path is not None:
            builder = builder.based_on_ckpt(self.args.pretrained_path).reinitialize(
                "nets.plainunet.PlainHead",
                output_channel = 118,
            )
        else:
            builder = builder.based_on_plan(
                "nets.ubimamba.UBiMamba",
                self.plan,
                input_channel=1,
                output_channel=118,
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

    def get_argument_parser(self):
        parser = ArgumentAdaptor.get_argument_parser(self)
        parser.add_argument("data", type=resolved_path)
        parser.add_argument("save_path", type=resolved_path)
        parser.add_argument("plan_path", type=resolved_path)
        parser.add_argument("-c", "--ckpt", required=True, type=resolved_path)
        parser.add_argument("-d", "--devices", type=int, default=[0], nargs="+")
        return parser

    def parse_args(self, args):
        ArgumentAdaptor.parse_args(self, args)
        self.data = self.args.data
        self.save_path = self.args.save_path / self.meta.lib
        if self.meta.experiment_name is not None:
            self.save_path = self.save_path / self.meta.experiment_name
        self.save_path = self.save_path / nowstring()
        self.plan = Plan(self.args.plan_path)
        self.devices = self.args.devices
        self.ckpt_path = self.args.ckpt

    def _build_data_module(self):
        return DataModule(self.data, self.plan, direct_test_dir=True)

    def _build_module(self):
        builder = Builder().based_on_ckpt(self.ckpt_path).to_params()
        lm = Model(builder, plan=self.plan)
        return lm


def inference(args, meta):
    PlainSegInferencer(args, meta)()
