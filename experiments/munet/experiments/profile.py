from experiments.munet import BottleneckSeg
from lightning.pytorch.profilers import AdvancedProfiler
from experiments.datamodule import DummyDataModule as DataModule


class Profile(BottleneckSeg):
    def _build_data_module(self):
        return DataModule(self.plan, lower=200, upper=400, dim=3)

    def configure_trainer(self, config):
        config = super().configure_trainer(config)
        config["limit_train_batches"] = 1
        config["profiler"] = AdvancedProfiler()
        return config


def train(args, parsed):
    Profile(args, parsed)()
