from pathlib import Path
from experiments.utils import nowstring


image_key = "image"
label_key = "label"


def default_loggers(save_path: Path, experiment_name: str):
    from lightning.pytorch.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger
    return [
        CSVLogger(save_path, name="training.log"),
        MLFlowLogger(
            experiment_name,
        ),
        TensorBoardLogger(save_path, name=experiment_name),
    ]


def default_callbacks(save_path: Path):
    from lightning.pytorch.callbacks import ModelCheckpoint

    return [
        ModelCheckpoint(
            save_path,
            "ckpt_{epoch}.pth",
            save_last=True,
            save_top_k=-1,
            every_n_epochs=10,
        )
    ]


def default_training_config(save_path: Path, meta, devices):
    return {
        "logger": default_loggers(
            save_path,
            (
                meta.lib
                if meta.experiment_name is None
                else meta.lib + meta.experiment_name
            ),
        ),
        "devices": devices,
        "max_epochs": 1000,
        "min_epochs": 1000,
        "limit_train_batches": 250,
        "default_root_dir": save_path,
        "precision": "bf16-mixed",
        "callbacks": default_callbacks(save_path),
        "accelerator": "gpu",
    }


assertion = False
