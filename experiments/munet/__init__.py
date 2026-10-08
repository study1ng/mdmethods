from experiments.nets.base import UNet, UNetReinitializer
from experiments.nets.builder import Builder
from experiments.pretrained_seg import PlainSegmentation, PlainSegInferencer, analyze, prune
from experiments.plan import Plan
import torch
from lightning.pytorch.callbacks import BaseFinetuning
from experiments.munet.datamodule import NoCropDataModule as DataModule
from experiments.munet.model import MUNetTrainingModule as Model
from copy import copy
import argparse
from experiments.config import image_key, label_key
from experiments.munet.evaluation import paired_cases, write_metrics
from experiments import ArgumentAdaptor
from experiments.utils import resolved_path

class BottleneckFinetuning(BaseFinetuning):
    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module: Model):
        self.freeze(pl_module.unet)
        for p in pl_module.unet.parameters():
            p.requires_grad_(False)

    def finetune_function(self, pl_module, epoch, optimizer):
        pass


class BottleneckSeg(PlainSegmentation):
    def configure_trainer(self, config):
        config = super().configure_trainer(config)
        config["callbacks"].append(BottleneckFinetuning())
        return config
    
    def _build_data_module(self):
        return DataModule(self.data, self.plan)

    def _build_module(self):
        builder = Builder()
        if self.args.pretrained_path is not None:
            builder = builder.based_on_ckpt(self.args.pretrained_path)
        else:
            raise Exception("Munet needs pretrained model")
        builder = builder.to_params()
        lm = Model(builder=builder, plan=self.plan, overlap_scale=0.25, gamma=1e-5)
        return lm


def train(args, parsed):
    BottleneckSeg(args, parsed)()


class MUNetInferencer(PlainSegInferencer):
    def _build_module(self):
        # Validate metadata before constructing the model or starting the trainer.
        # Plan and builder objects require loading the full, trusted checkpoint.
        checkpoint = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("Expected a MUNet Lightning checkpoint.")
        state = checkpoint.get("state_dict", {})
        if not any(key.startswith("bottleneck.") for key in state):
            raise ValueError("The checkpoint has no MUNet bottleneck weights.")
        hparams = checkpoint.get("hyper_parameters", {})
        required = (
            "builder", "plan", "overlap_scale",
            "global_positional_encoding_proposition", "gamma",
        )
        missing = [key for key in required if key not in hparams]
        if missing:
            raise ValueError(f"MUNet checkpoint is missing hyperparameters: {missing}")
        saved_plan = hparams["plan"]
        if not isinstance(saved_plan, Plan):
            raise ValueError("The checkpoint must contain the training Plan object.")
        # Compare effective settings, not file paths or batch size.
        plan_fields = (
            "patch_size", "spacing", "mean", "std",
            "percentile_00_5", "percentile_99_5",
            "pool_strides", "conv_kernel_size", "stem_channel",
            "max_feature_channel", "n_stages", "dim",
        )
        mismatched = [
            field for field in plan_fields
            if not hasattr(saved_plan, field)
            or getattr(saved_plan, field) != getattr(self.plan, field)
        ]
        if mismatched:
            raise ValueError(
                f"Inference plan differs from the training plan: {mismatched}. "
                "Use the plan used to train this checkpoint."
            )
        # Release checkpoint tensors before Lightning reads them for restoration.
        del state, hparams, saved_plan, checkpoint
        try:
            return Model.load_from_checkpoint(
                self.ckpt_path, map_location="cpu", strict=True,
                weights_only=False, plan=self.plan,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "MUNet restoration needs the checkpoints referenced by its saved "
                "builder. Make the original pretrained checkpoints available at "
                f"their recorded paths. Missing file: {exc.filename or exc}"
            ) from exc


def inference(args, meta):
    MUNetInferencer(args, meta)()

class MUNetCustom(MUNetInferencer):
    def get_argument_parser(self):
        parser = argparse.ArgumentParser()
        actions = parser.add_subparsers(dest="action", required=True)
        val_parser = actions.add_parser(
            "val", parents=[super().get_argument_parser()], add_help=False
        )
        _add_metric_arguments(val_parser)
        return parser

    @staticmethod
    def _hd_percentile(value):
        percentile = float(value)
        # MONAI treats zero as an unspecified percentile (maximum distance).
        if not 0 < percentile <= 100:
            raise argparse.ArgumentTypeError("--hd must be greater than 0 and at most 100")
        return percentile

    def parse_args(self, args):
        super().parse_args(args)
        self.dataset_root = self.data
        self.cases = paired_cases(
            self.dataset_root / image_key, self.dataset_root / label_key
        )
        self.data = self.dataset_root / image_key
        if self.save_path.exists():
            raise FileExistsError(f"Output directory already exists: {self.save_path}")

    def _evaluation_num_classes(self):
        head = self.module.unet.decoder.head
        heads = list(head) if isinstance(head, torch.nn.ModuleList) else [head]
        channels = [getattr(item, "output_channel", None) for item in heads]
        if not channels or any(
            not isinstance(channel, int) or channel < 2 for channel in channels
        ):
            raise ValueError(
                "Evaluation requires output heads with at least one foreground "
                f"class (including background, channels={channels})."
            )
        if len(set(channels)) != 1:
            raise ValueError(f"Output heads have inconsistent class counts: {channels}")
        return channels[0]

    def __call__(self):
        # Reinitializing a head may leave unet.output_channel at its old value.
        # Read the restored heads and validate before starting expensive inference.
        num_classes = self._evaluation_num_classes()
        print(f"Inference and metrics output: {self.save_path}")
        original_meta = self.meta
        self.meta = copy(self.meta)
        self.meta.method = "inference"
        try:
            result = super().__call__()
        finally:
            self.meta = original_meta
        # All distributed workers must finish writing before evaluation reads files.
        self.trainer.strategy.barrier()
        if self.trainer.is_global_zero:
            write_metrics(
                self.cases, self.save_path,
                epoch=self.trainer.current_epoch,
                num_classes=num_classes,
                dice=self.args.dice, hd=self.args.hd,
            )
        return result


def _add_metric_arguments(parser):
    parser.add_argument("--dice", action="store_true")
    parser.add_argument("--hd", type=MUNetCustom._hd_percentile, default=None)


class MUNetMetric(ArgumentAdaptor):
    def get_argument_parser(self):
        parser = super().get_argument_parser()
        parser.add_argument("data", type=resolved_path)
        parser.add_argument("prediction_path", type=resolved_path)
        _add_metric_arguments(parser)
        return parser

    def __call__(self):
        cases = paired_cases(self.args.data / image_key, self.args.data / label_key)
        write_metrics(
            cases, self.args.prediction_path,
            dice=self.args.dice, hd=self.args.hd,
        )


def custom(args, meta):
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["val", "metric"])
    parser.add_argument("-mh", "--module-help", action="help")
    # Leave subcommand help and arguments to the corresponding adaptor.
    if args and args[0] in {"val", "metric"}:
        action, remaining = args[0], args[1:]
    else:
        parsed, remaining = parser.parse_known_args(args)
        action = parsed.action
    if action == "metric":
        MUNetMetric(remaining, meta)()
    else:
        MUNetCustom(["val", *remaining], meta)()
