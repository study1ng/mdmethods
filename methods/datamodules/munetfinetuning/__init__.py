from pathlib import Path
import numpy as np

import lightning as L
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    RandZoomd,
    RandRotated,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandSimulateLowResolutiond,
    RandScaleIntensityd,
    RandAdjustContrastd,
    MaskIntensityd,
)

from preprocess import load_transformd, planned_transformd


def augmentation_transforms(
    plan, image_key: list = ["image"], label_key: list | None = ["label"]
):
    all_key = image_key + label_key if label_key is not None else image_key
    need_label = label_key is not None
    patch_size = plan["configurations"]["3d_fullres"]["patch_size"]
    do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > 3
    if do_dummy_2d_data_aug:
        rotation_for_DA = {
            "range_x": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
            "range_y": (0, 0),
            "range_z": (0, 0),
        }
        min_zoom = [0.7, 1.0, 1.0]
        max_zoom = [1.4, 1.0, 1.0]
    else:
        rotation_for_DA = {
            "range_x": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
            "range_y": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
            "range_z": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
        }
        min_zoom = 0.7
        max_zoom = 1.4
    interp_modes = ["bilinear"] * len(image_key)
    if need_label:
        interp_modes += ["nearest"] * len(label_key)
    composelist = [
        load_transformd(all_key),
        planned_transformd(plan, image_key, label_key),
        RandRotated(all_key, **rotation_for_DA, prob=0.2),
        RandZoomd(all_key, prob=0.2, min_zoom=min_zoom, max_zoom=max_zoom, mode=interp_modes),
        RandRotated(all_key, **rotation_for_DA, prob=0.2, mode=interp_modes),
        RandGaussianNoised(image_key, prob=0.1),
        RandGaussianSmoothd(
            image_key,
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
            prob=0.2,
        ),
        RandScaleIntensityd(
            image_key, factors=(0.75, 1.25), prob=0.15, channel_wise=True
        ),
        RandAdjustContrastd(
            image_key, prob=0.1, gamma=(0.7, 1.5), invert_image=True, retain_stats=True
        ),
        RandSimulateLowResolutiond(
            image_key,
            prob=0.25,
            downsample_mode="nearest",
            upsample_mode="trilinear",
            zoom_range=(0.5, 1.0),
        ),
    ]
    need_label and composelist.append(MaskIntensityd(image_key, mask_key=label_key[0]))
    composelist += [
        RandFlipd(all_key, prob=0.5, spatial_axis=0),
        RandFlipd(all_key, prob=0.5, spatial_axis=1),
        RandFlipd(all_key, prob=0.5, spatial_axis=2),
    ]
    return Compose(composelist)


class MUNetFinetuningDataModule(L.LightningDataModule):
    def __init__(
        self,
        preprocessed_dir: str | Path,
        dataset_name: str,
        plan,
        num_workers: int = 4,
    ):
        super().__init__()
        self.preprocessed_dir = Path(preprocessed_dir)
        self.dataset_name = dataset_name
        self.plan = plan
        self.num_workers = num_workers
        self.keys = ["image", "label"]
        self.img_key = ["image"]
        self.batch_size = self.plan["configurations"]["3d_fullres"]["batch_size"]

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            pimgs = self.preprocessed_dir / self.dataset_name / "images"
            plabels = self.preprocessed_dir / self.dataset_name / "labels"
            assert pimgs.exists(), f"the preprocessed img dir {pimgs} do not exists"
            assert (
                plabels.exists()
            ), f"the preprocessed label dir {plabels} do not exists"
            pimgs_files = list(pimgs.iterdir())
            pimgs_files = list(sorted(pimgs_files, key=str))
            plabels_files = list(plabels.iterdir())
            plabels_files = list(sorted(plabels_files, key=str))
            stem = lambda img: img.name.split(".")[0].split("_")[0]
            assert all(
                stem(pimg) == stem(plabel)
                for pimg, plabel in zip(pimgs_files, plabels_files, strict=True)
            )

            files = [
                {
                    "image": pimg,
                    "label": plabel,
                    "name": pimg.name.split(".")[0].split("_")[0],
                }
                for pimg, plabel in zip(pimgs_files, plabels_files, strict=True)
                if pimg.suffix == ".gz" and plabel.suffix == ".gz"
            ]

            transforms = augmentation_transforms(self.keys, self.img_key, self.plan)
            self.train_dataset = Dataset(files, transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
