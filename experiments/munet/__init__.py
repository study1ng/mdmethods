# from experiments.prune import NoPruner as Pruner
# from experiments.munet.preprocess import PlannedPreprocessor as Preprocessor
# from experiments.utils import resolved_path
# from pathlib import Path
# from experiments.munet.datamodule import NoCropDataModule as DataModule
# from experiments.munet.model import MUNet as Model
# from experiments.nets import UBiMamba as Net
# import argparse, lightning as L
# from lightning.pytorch.loggers import CSVLogger
# from lightning.pytorch.callbacks import ModelCheckpoint
# from experiments.analyze import CTAnalyzer as Analyzer
# import torch
# from experiments.plan import Plan


# def prune(args):
#     Pruner(args)()


# def analyze(args):
#     Analyzer(args)()


# def preprocess(args):
#     Preprocessor(args)()


# def train(args):
#     torch.set_float32_matmul_precision("medium")
#     args = _train_argparse(args)
#     _train(
#         preprocessed=args.preprocessed,
#         pretrained=args.pretrained,
#         plan_path=args.plan_path,
#         device=args.devices,
#         checkpoint=args.ckpt,
#         maximum_gpu_memory_limit=args.maximum_gpu_memory_limit,
#     )


# def _train_argparse(args):
#     parser = argparse.ArgumentParser()
#     parser.add_argument("preprocessed", type=resolved_path)
#     parser.add_argument("pretrained", type=resolved_path)
#     parser.add_argument("plan_path", type=resolved_path)
#     parser.add_argument("-d", "--devices", type=int, default=[0], nargs="+")
#     parser.add_argument("-c", "--ckpt", default=None, type=resolved_path)
#     parser.add_argument(
#         "-m",
#         "--maximum_gpu_memory_limit",
#         default=5.0,
#         type=float,
#         help="maximum gpu memory limit for feature maps in gb",
#     )
#     args = parser.parse_args(args)
#     return args


# def _train(
#     preprocessed: Path,
#     pretrained: Path,
#     plan_path: str,
#     maximum_gpu_memory_limit: float,
#     device: list[int] = [0],
#     checkpoint: str | None = None,
# ):
#     plan = Plan(plan_path)
#     pretrained.mkdir(parents=True, exist_ok=True)
#     if checkpoint is not None:
#         checkpoint = Path(checkpoint)
#     dm = DataModule(preprocessed, plan)
#     lm = Net.from_plan(plan.plan, 1, 118)
#     munet = Model(lm, maximum_gpu_memory_limit * (1 << 30))
#     tr = L.Trainer(
#         logger=[CSVLogger(pretrained, name="pretraining.log")],
#         devices=device,
#         max_epochs=1000,
#         min_epochs=1000,
#         limit_train_batches=250,
#         default_root_dir=pretrained,
#         precision="bf16-mixed",
#         callbacks=[
#             ModelCheckpoint(
#                 pretrained,
#                 "ckpt_{epoch}.pth",
#                 save_last=True,
#                 save_top_k=-1,
#                 every_n_epochs=10,
#             )
#         ],
#         accelerator="gpu",
#         accumulate_grad_batches=2,
#     )
#     tr.fit(model=munet, datamodule=dm, ckpt_path=checkpoint)
