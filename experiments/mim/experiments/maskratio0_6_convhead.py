import torch
import torch.nn.functional as F

from experiments.config import image_key
from experiments.mim.datamodule import SSLDataModule as DataModule
from experiments.mim.model import MIMModule as Model
from experiments.nets.builder import Builder
from experiments.trainer import PlannedExperiment
from experiments.nets.ubimamba import UBiMamba as UNet
from experiments.prune import SpacingShapeStrictPruner as Pruner
from experiments.analyze import CTAnalyzer as Analyzer
from experiments.mim import inference
from experiments.utils import elementwise_mul, is_integer, denominator


class MIMModule(Model):
    def __init__(
        self,
        *args,
        target_downsample_mode: str | None = None,
        mask_downsample_mode: str = "nearest",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_downsample_mode = (
            target_downsample_mode or self._default_downsample_mode()
        )
        self.mask_downsample_mode = mask_downsample_mode

    def _default_downsample_mode(self) -> str:
        match self.unet.dim:
            case 1:
                return "linear"
            case 2:
                return "bilinear"
            case 3:
                return "trilinear"
            case _:
                raise ValueError(f"unsupported dim: {self.unet.dim}")

    @staticmethod
    def _interpolate(x: torch.Tensor, size: tuple[int, ...], mode: str) -> torch.Tensor:
        if tuple(x.shape[2:]) == tuple(size):
            return x
        if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            return F.interpolate(x, size=size, mode=mode, align_corners=False)
        return F.interpolate(x, size=size, mode=mode)

    def _target_for_prediction(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        index: int,
    ) -> torch.Tensor:
        return self._interpolate(
            target,
            tuple(pred.shape[2:]),
            self.target_downsample_mode,
        )

    def _mask_for_prediction(
        self,
        mask: torch.Tensor,
        pred: torch.Tensor,
        index: int,
    ) -> torch.Tensor:
        return self._interpolate(
            mask,
            tuple(pred.shape[2:]),
            self.mask_downsample_mode,
        )

    def _training_loss_components(self, out, target, mask):
        """Compute loss per prediction scale because each ConvHead output can
        have a different spatial size.
        """
        outputs = self._iter_outputs(out)
        loss = target.new_tensor(0.0)
        visible_loss = target.new_tensor(0.0)
        masked_loss = target.new_tensor(0.0)

        for i, pred in enumerate(outputs):
            target_i = self._target_for_prediction(target, pred, i)
            mask_i = self._mask_for_prediction(mask, pred, i)
            loss_i, visible_i, masked_i = self._reduce_masked_visible_loss(
                elementwise_loss=self.loss(pred, target_i),
                mask=mask_i,
                channel_count=pred.shape[1],
            )
            weight = self._output_weight(i, len(outputs))
            loss = loss + weight * loss_i
            visible_loss = visible_loss + weight * visible_i
            masked_loss = masked_loss + weight * masked_i

        return loss, visible_loss, masked_loss


def prune(args, meta):
    Pruner(args, meta)()


def analyze(args, meta):
    Analyzer(args, meta)()


class MIM(PlannedExperiment):
    """Masked Image Modeling"""

    def __init__(self, args, parsed):
        super().__init__(args, parsed)
        torch.set_float32_matmul_precision("medium")

    def _build_data_module(self):
        return DataModule(self.data, self.plan)

    def _build_module(self):
        builder = (
            Builder()
            .based_on_plan(
                "nets.ubimamba.UBiMamba",
                self.plan,
                input_channel=1,
                output_channel=1,
                deep_supervision=True,
            )
            .to_params()
        )
        lm = MIMModule(builder, mask_ratio=0.6, head=None)
        return lm


def train(args, meta):
    MIM(args, meta)()
