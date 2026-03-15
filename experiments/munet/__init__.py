from ..prune import SpacingShapeStrictPruner
from ..preprocess import PlannedSSLPreprocessor
import argparse

def prune(args):
    SpacingShapeStrictPruner(args)()

def preprocess(args):
    PlannedSSLPreprocessor(args)()

def train(args):
    args = _train_argparse(args)
    _train(args.preprocessed, args.pretrained, args.dataset, args.plan_path, args.devices, args.ckpt)

def _train_argparse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("preprocessed", type=resolved_path)
    parser.add_argument("pretrained", type=resolved_path)
    parser.add_argument("dataset")
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
    device: list[int] = [0],
    checkpoint: str | None = None,
):
    plan = json.load(Path(plan_path).open())
    ptdir = pretrained / dataset / nowstring()
    ptdir.mkdir(parents=True, exist_ok=True)
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
    dm = MUNetFinetuningDataModule(preprocessed, dataset, plan)
    lm = UMambaEnc.from_plan(plan, 1, 10)
    munet = MUNet(lm, 40 << 20)
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
        accumulate_grad_batches=2,
    )
    tr.fit(model=munet, datamodule=dm, ckpt_path=checkpoint)
