from experiments.nets.base import UNet, UNetReinitializer
from experiments.nets.builder import Builder
from experiments.pretrained_seg import PlainSegmentation, PlainSegInferencer, analyze, prune
from experiments.plan import Plan
import torch
from lightning.pytorch.callbacks import BaseFinetuning
from experiments.munet.datamodule import NoCropDataModule as DataModule
from experiments.munet.model import MUNetTrainingModule as Model

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
