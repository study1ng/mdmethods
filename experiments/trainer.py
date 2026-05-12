from pprint import pprint
from lightning import LightningDataModule, Trainer
from experiments import ArgumentAdaptor
from abc import abstractmethod
from experiments.nets.builder import Builder
from experiments.plan import Plan
from experiments.utils import nowstring, resolved_path
from experiments.assertions import AssertEq
import lightning as L
from pathlib import Path
from experiments.nets.base import UNet
import torch
from torch import tensor, Tensor
from experiments.callbacks import LogCallback
from experiments.config import default_training_config


class UNetTrainingModule(L.LightningModule):
    head_weights: Tensor
    CKPT_BUILDER_KEY: str = "_mm_builder"

    def __init__(self, builder: dict, weights):
        super().__init__()
        print("using builder: ")
        pprint(builder)
        self.builder = builder
        self.unet = Builder.from_params(builder).build()
        print(self.unet)
        self.deep_supervision = self.unet.deep_supervision
        weights = self.initialize_weights(weights)
        self.register_buffer("head_weights", weights)

    def initialize_weights(self, w, *, default=0.5):
        if self.deep_supervision:
            if isinstance(w, tuple):
                AssertEq()(len(self.unet.decoder.head), len(w))
            if w is None:
                w = default
            if isinstance(w, float):
                w = tuple(w**i for i in range(self.unet.n_stages + 1))
            w = tensor(w, dtype=torch.float32)
            w /= w.sum()
        return w

    def on_save_checkpoint(self, checkpoint):
        checkpoint[self.CKPT_BUILDER_KEY] = self.builder

def dl_pretrained_unet(
    pretrained_module_path: Path | str,
) -> UNet:
    """load from pretrained checkpoint

    Parameters
    ----------
    pretrained_module_path : Path | str
        the path of UNetTrainingModule checkpoint path

    Returns
    -------
    UNet
        pretrained unet
    """
    checkpoint = torch.load(pretrained_module_path, map_location="cpu")
    builder = Builder.from_params(checkpoint[UNetTrainingModule.CKPT_BUILDER_KEY])
    print("pretrained unet recipe: ")
    pprint(builder.actions)
    unet = builder.build()
    state_dict = checkpoint["state_dict"]
    unet_state_dict = {
        k.removeprefix("unet."): v
        for k, v in state_dict.items()
        if k.startswith("unet.")
    }
    unet.load_state_dict(unet_state_dict, strict=True)
    return unet


class Experiment(ArgumentAdaptor):
    def _build_trainer(self) -> Trainer:
        config = default_training_config(
            save_path=self.save_path, meta=self.meta, devices=self.devices
        )
        config = self.configure_trainer(config)
        print("trainer config: ", config)
        tr = Trainer(**config)
        return tr

    def configure_trainer(self, config):
        return config
    
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





class Inferencer(ArgumentAdaptor):
    def __init__(self, args, parsed):
        super().__init__(args, parsed)
        self.datamodule = self._build_data_module()
        self.module = self._build_module()
        self.trainer = self._build_trainer()

    def get_argument_parser(self):
        parser = super().get_argument_parser()
        parser.add_argument("preprocessed", type=resolved_path)
        parser.add_argument("save_path", type=resolved_path)
        parser.add_argument("ckpt_path", type=resolved_path)
        parser.add_argument("-d", "--devices", type=int, default=[0], nargs="+")
        return parser

    @abstractmethod
    def _build_trainer(self) -> Trainer: ...

    def parse_args(self, args):
        super().parse_args(args)
        self.preprocessed = self.args.preprocessed
        self.save_path = self.args.save_path / self.meta.lib
        if self.meta.experiment_name is not None:
            self.save_path = self.save_path / self.meta.experiment_name
        self.save_path = self.save_path / nowstring()
        self.devices = self.args.devices
        self.ckpt_path = self.args.ckpt_path

    @abstractmethod
    def _build_module(self) -> UNetTrainingModule: ...

    @abstractmethod
    def _build_data_module(self) -> LightningDataModule: ...

    def __call__(self):
        match self.meta.method:
            case "inference":
                return self.trainer.test(model=self.module, datamodule=self.datamodule)
            case _:
                raise NotImplementedError(f"{self.meta.method} is not implemented")


class PlannedInferencer(Inferencer):
    def get_argument_parser(self):
        parser = super().get_argument_parser()
        parser.add_argument("plan_path", type=resolved_path)
        return parser

    def _build_trainer(self):
        tr = Trainer(
            callbacks=[LogCallback(self.save_path)], devices=self.devices
        )
        return tr

    def parse_args(self, args):
        super().parse_args(args)
        self.plan = Plan(self.args.plan_path)
