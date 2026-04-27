import json
from pathlib import Path
import numpy as np

import lightning as L
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    RandZoomd,
    RandRotated,
    RandFlipd,
)
from experiments.config import image_key
from experiments.preprocess import (
    load_transformd,
    padded_crop_wrapper,
    planned_transformd,
)
from experiments.plan import Plan


def augmentation_transforms(keys, plan: Plan):
    patch_size = plan.patch_size
    do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > 3
    if do_dummy_2d_data_aug:
        rotation_for_DA = {
            "range_x": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
            "range_y": (0, 0),
            "range_z": (0, 0),
        }
        min_zoom = [0.8, 1.0, 1.0]
        max_zoom = [1.2, 1.0, 1.0]
    else:
        rotation_for_DA = {
            "range_x": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
            "range_y": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
            "range_z": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
        }
        min_zoom = 0.8
        max_zoom = 1.2

    composelist = [
        load_transformd(keys),
        planned_transformd(plan, (image_key,)),
        padded_crop_wrapper(
            keys,
            patch_size,
            [
                RandRotated(keys, **rotation_for_DA, prob=0.2),
                RandZoomd(keys, prob=0.2, min_zoom=min_zoom, max_zoom=max_zoom),
                RandFlipd(keys, prob=0.5, spatial_axis=0),
                RandFlipd(keys, prob=0.5, spatial_axis=1),
                RandFlipd(keys, prob=0.5, spatial_axis=2),
            ],
        ),
    ]
    return Compose(composelist)


class SSLDataModule(L.LightningDataModule):
    """Data Module for Self supervised learning which don't need label
    """
    def __init__(
        self,
        preprocessed: str | Path,
        plan: Plan,
        num_workers: int = 4,
    ):
        super().__init__()
        self.preprocessed = Path(preprocessed)
        self.plan = plan
        self.num_workers = num_workers
        self.keys = [image_key]
        self.batch_size = self.plan.batch_size

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            assert self.preprocessed.exists(), f"the preprocessed img dir {self.preprocessed} do not exists"

            imgs = [
                {image_key: img, "name": img.name.split(".")[0].split("_")[0]}
                for img in self.preprocessed.iterdir()
                if img.suffix == ".gz"
            ]

            transforms = augmentation_transforms(self.keys, self.plan)
            self.train_dataset = Dataset(imgs, transforms)

        if stage == "test":
            assert self.preprocessed.exists(), f"the preprocessed img dir {self.preprocessed} do not exists"

            imgs = [
                {image_key: img, "name": img.name.split(".")[0].split("_")[0]}
                for img in self.preprocessed.iterdir()
                if img.suffix == ".gz"
            ]

            transforms = augmentation_transforms(self.keys, self.plan)
            self.test_dataset = Dataset(imgs, transforms)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )


    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
