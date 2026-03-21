from gc import callbacks

from lightning import LightningDataModule, LightningModule, Trainer

from experiments import ArgumentAdaptor
from abc import abstractmethod
from experiments.config import default_training_config
from experiments.plan import Plan
from experiments.utils import nowstring, resolved_path


class Experiment(ArgumentAdaptor):
    def _build_trainer(self) -> Trainer:
        config = default_training_config(
            save_path=self.save_path, meta=self.meta, devices=self.devices
        )
        print("trainer config: ", config)
        tr = Trainer(**config)
        return tr

    @abstractmethod
    def _build_module(self) -> LightningModule: ...

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
                raise NotImplemented(f"{self.meta.method} is not implemented")


class PlannedExperiment(Experiment):
    def get_argument_parser(self):
        parser = super().get_argument_parser()
        parser.add_argument("plan_path", type=resolved_path)
        return parser

    def parse_args(self, args):
        super().parse_args(args)
        self.plan = Plan(self.args.plan_path)
