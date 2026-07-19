from experiments.nets.builder import Builder
from experiments.trainer import PlannedExperiment, PlannedInferencer
from experiments.prune import NoPruner as Pruner
from experiments.analyze import CTAnalyzer as Analyzer
from experiments.pretrained_seg.datamodule import CropSegDataModule as DataModule
from experiments.pretrained_seg.model import SegmentationModule as Model
import torch
from experiments.utils.fsutils import resolved_path


def prune(args, meta):
    Pruner(args, meta)()


def analyze(args, meta):
    Analyzer(args, meta)()