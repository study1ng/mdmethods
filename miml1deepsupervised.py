from methods.structures.mim import UNetEncoderWithMIM
from methods.structures.nets.UBiMambaEnc_3d import UMambaEnc
from methods.datamodules.mim import SSLDataModule
from pathlib import Path
import json
from utils import nowstring, resolved_path
import torch
torch.set_float32_matmul_precision("medium")
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint


def main(
    preprocessed: Path,
    pretrained: Path,
    dataset: str,
    plan_path: str,
    device: int = 0,
    checkpoint: str | None = None,
):
    plan = json.load(Path(plan_path).open())
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("preprocessed", type=resolved_path)
    parser.add_argument("pretrained", type=resolved_path)
    parser.add_argument("dataset")
    parser.add_argument("plan_path", type=resolved_path)
    parser.add_argument("-d", "--devices", type=int, default=[0], nargs="+")
    parser.add_argument("-c", "--ckpt", default=None, type=resolved_path)
    args = parser.parse_args()
    main(args.preprocessed, args.pretrained, args.dataset, args.plan_path, args.devices, args.ckpt)