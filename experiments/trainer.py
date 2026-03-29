from lightning import LightningDataModule, Trainer
from experiments import ArgumentAdaptor
from abc import abstractmethod
from experiments.config import default_training_config
from experiments.plan import Plan
from experiments.utils import nowstring, resolved_path
from experiments.assertions import AssertEq
import importlib
import warnings
import lightning as L
from pathlib import Path
from experiments.nets.base import UNet, UNetHead
import torch
from torch import tensor, Tensor


class UNetTrainingModule(L.LightningModule):
    head_weights: Tensor
    def __init__(self, unet: UNet, *, weights):
        super().__init__()
        self.unet = unet
        self.deep_supervision = unet.deep_supervision
        self.save_hyperparameters(ignore=["unet"])
        if hasattr(unet, "hparams"):
            self.save_hyperparameters({"unet_hparams": dict(unet.hparams)})
        weights = self.initialize_weights(weights)
        self.register_buffer("head_weights", weights)

    def initialize_weights(self, w, *, default = 0.5):
        if self.deep_supervision:
            if isinstance(w, tuple):
                AssertEq()(len(self.unet.decoder.head), len(w))
            if w is None:
                w = default
            if isinstance(w, float):
                w = tuple(
                    w**i for i in range(self.unet.n_stages + 1)
                )
            w = tensor(w, dtype=torch.float32)
            w /= w.sum()
        return w


def load_from_pretrained(
    pretrained_path: Path | str,
    unet: UNet | None = None,
) -> tuple[UNet, UNetHead]:
    """load from pretrained checkpoint

    Parameters
    ----------
    pretrained_module_path : Path | str
        the path of UNetTrainingModule checkpoint path

    unet : UNet
        a unet, if assigned, it will be used for loading state dict

    Returns
    -------
    UNet
        pretrained unet
    UNetHead
        original unet head
    """
    checkpoint = torch.load(pretrained_path, map_location="cpu")
    hparams = checkpoint.get("hyper_parameters")
    if unet is None:  # unetがないならunet hparamsがなければならない
        assert (
            hparams is not None
        ), "unet was not assigned and no hyper parameters was saved"
        unet_hparams = hparams.get("unet_hparams")
        assert (
            unet_hparams is not None
        ), "unet was not assigned and no unet hparams was saved"
        unet_name = unet_hparams.get("_unetname")
        if not unet_name:
            raise ValueError(
                "due to unet was not assigned, we tried to load unet from hyperparameters but no unet name there"
            )
        if "." in unet_name:
            unet_module, unet_clsname = unet_name.rsplit(".", 1)
        else:
            unet_module = "experiments.nets"
            unet_clsname = unet_name
        try:
            unet_cls = getattr(importlib.import_module(unet_module), unet_clsname)
        except Exception as e:
            raise ValueError(
                f"due to unet was not assigned, we tried to load unet from hyperparameters. \
it gave us {unet_name} but we could not found unet there cuz {e}"
            )
        unet_args = {k: v for k, v in unet_hparams.items() if not k.startswith("_")}
        unet = unet_cls(**unet_args)

    original_head = unet.decoder.head

    state_dict = checkpoint.get("state_dict", {})
    unet_state_dict = {
        k.removeprefix("unet."): v
        for k, v in state_dict.items()
        if k.startswith("unet.")
    }

    def _get_pretrained_head_and_params(hparams):
        if hparams is None:
            warnings.warn("no hyperparameters was saved")
            return None, None
        unet_hparams = hparams.get("unet_hparams")
        if unet_hparams is None:
            warnings.warn("no unet hparams was found")
            return None, None
        args = unet_hparams.get("_headparams")
        if args is None:
            print(
                f"no head params found in checkpoint so we'll treat it as no arguments"
            )
            args = ((), {})
        return unet_hparams.get("_headname"), args

    pretrained_head, args = _get_pretrained_head_and_params(hparams)
    if pretrained_head is None:
        warnings.warn(
            f"due to we could not found head name from checkpoint, we'll use the default head {type(unet.decoder.head)}"
        )
        pretrained_head_class = None
    else:
        if "." in pretrained_head:
            head_module, head_clsname = pretrained_head.rsplit(".", 1)
        else:
            raise ValueError(
                f"We can't specify the pretrained head loc from {pretrained_head}"
            )
        try:
            pretrained_head_class: type = getattr(
                importlib.import_module(head_module), head_clsname
            )
        except Exception as e:
            warnings.warn(f"we could not load {head_module}.{head_clsname} due to {e}")
            pretrained_head_class = None
        pretrained_headargs, pretrained_headkwargs = args

    print(
        f"we'll use unet class {type(unet).__name__}, head class {pretrained_head_class.__name__ if pretrained_head_class is not None else type(unet.decoder.head).__name__}"
    )

    if pretrained_head_class is None:
        warnings.warn(
            "no pretrained head was found so we'll try to use unstrict load_state_dict"
        )
        unet.load_state_dict(unet_state_dict, strict=False)
    else:
        unet = pretrained_head_class.attach_to_unet(
            unet, *pretrained_headargs, **pretrained_headkwargs
        )
        unet.load_state_dict(unet_state_dict, strict=True)

    return unet, original_head


class Experiment(ArgumentAdaptor):
    def _build_trainer(self) -> Trainer:
        config = default_training_config(
            save_path=self.save_path, meta=self.meta, devices=self.devices
        )
        print("trainer config: ", config)
        tr = Trainer(**config)
        return tr

    @abstractmethod
    def _build_module(self) -> UNetTrainingModule: ...

    @abstractmethod
    def _build_data_module(self) -> LightningDataModule: ...

    def __init__(self, args, parsed):
        super().__init__(args, parsed)
        self.datamodule = self._build_data_module()
        self.module = self._build_module()
        self.trainer = self._build_trainer()

    def get_argument_parser(self):
        parser = super().get_argument_parser()
        parser.add_argument("data", type=resolved_path)
        parser.add_argument("save_path", type=resolved_path)
        parser.add_argument("-d", "--devices", type=int, default=[0], nargs="+")
        parser.add_argument("-c", "--ckpt", default=None, type=resolved_path)
        return parser

    def parse_args(self, args):
        super().parse_args(args)
        self.data = self.args.data
        self.save_path = self.args.save_path / self.meta.lib
        if self.meta.experiment_name is not None:
            self.save_path = self.save_path / self.meta.experiment_name
        self.save_path = self.save_path / nowstring()
        self.devices = self.args.devices
        self.ckpt = self.args.ckpt

    def __call__(self):
        match self.meta.method:
            case "train":
                return self.trainer.fit(
                    model=self.module, datamodule=self.datamodule, ckpt_path=self.ckpt
                )
            case "inference":
                assert self.ckpt is not None, "inference needs checkpoint path"
                return self.trainer.test(
                    model=self.module, datamodule=self.datamodule, ckpt_path=self.ckpt
                )
            case _:
                raise NotImplementedError(f"{self.meta.method} is not implemented")


class PlannedExperiment(Experiment):
    def get_argument_parser(self):
        parser = super().get_argument_parser()
        parser.add_argument("plan_path", type=resolved_path)
        return parser

    def parse_args(self, args):
        super().parse_args(args)
        self.plan = Plan(self.args.plan_path)
