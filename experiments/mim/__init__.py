from experiments.mim.preprocess import PlannedSSLPreprocessor
import argparse, lightning as L
from experiments.mim.datamodule import SSLDataModule
from experiments.mim.model import UNetEncoderWithMIM
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch

from experiments.utils import resolved_path, loaded_json, nowstring
from pathlib import Path
from experiments.nets.UBiMambaEnc_3d import UMambaEnc
from experiments.prune import SpacingShapeStrictPruner
from experiments.analyze import CTAnalyzer
from experiments.plan import Plan


def prune(args):
    SpacingShapeStrictPruner(args)()

def analyze(args):
    CTAnalyzer(args)()

def preprocess(args):
    PlannedSSLPreprocessor(args)()


def _train_argparse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("preprocessed", type=resolved_path)
    parser.add_argument("pretrained", type=resolved_path)
    parser.add_argument("dataset", type=str)
    parser.add_argument("plan_path", type=resolved_path)
    parser.add_argument("-d", "--devices", type=int, default=[0], nargs="+")
    parser.add_argument("-c", "--ckpt", default=None, type=resolved_path)
    args = parser.parse_args(args)
    return args


def _train(
    preprocessed: Path,
    pretrained: Path,
    dataset: str,
    plan_path: str,
    device: int = 0,
    checkpoint: str | None = None,
):
    torch.set_float32_matmul_precision("medium")
    plan = Plan(plan_path)
    ptdir = pretrained / dataset / nowstring()
    ptdir.mkdir(parents=True, exist_ok=True)
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
    dm = SSLDataModule(preprocessed, dataset, plan)
    lm = UNetEncoderWithMIM.from_plan(plan, UMambaEnc, 0.6, deep_supervision=True)
    tr = L.Trainer(
        logger=[CSVLogger(ptdir, name="pretraining.log")],
        devices=device,
        max_epochs=1000,
        min_epochs=1000,
        limit_train_batches=250,
        default_root_dir=ptdir,
        precision="bf16-mixed",
        callbacks=[
            ModelCheckpoint(
                ptdir,
                "ckpt_{epoch}.pth",
                save_last=True,
                save_top_k=-1,
                every_n_epochs=10,
            )
        ],
        accelerator="gpu",
    )
    tr.fit(model=lm, datamodule=dm, ckpt_path=checkpoint)


def train(args):
    args = _train_argparse(args)
    _train(
        args.preprocessed,
        args.pretrained,
        args.dataset,
        args.plan_path,
        args.devices,
        args.ckpt,
    )
