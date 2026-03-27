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
    SpatialPadd,
)
from experiments.config import filekey, image_key, label_key
from experiments.preprocess import (
    load_transformd,
    planned_transformd,
    padded_crop_wrapper,
)
from experiments.plan import Plan
from experiments.assertions import AssertEq


def augmentation_transforms(plan: Plan, image_key, label_key: list | None):
    all_key = image_key + label_key if label_key is not None else image_key
    need_label = label_key is not None
    patch_size = plan.patch_size
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
    ]
    with_croplist = [
        RandZoomd(
            all_key,
            prob=0.2,
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            mode=interp_modes,
            keep_size=True,
        ),
        RandRotated(all_key, **rotation_for_DA, prob=0.2, mode=interp_modes),
    ]
    composelist.append(
        padded_crop_wrapper(
            keys=all_key, crop_size=patch_size, transforms=with_croplist
        )
    )
    composelist += [
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
        # SpatialPadd(keys=all_key, spatial_size=patch_size, mode="replicate"), # UNetの入力サイズ固定されてないしパディング必要ないことに気づいた
    ]
    return Compose(composelist)


class CropSegDataModule(L.LightningDataModule):
    def __init__(
        self,
        preprocessed_dir: str | Path,
        plan: Plan,
        num_workers: int = 4,
    ):
        super().__init__()
        self.preprocessed_dir = Path(preprocessed_dir)
        self.plan = plan
        self.num_workers = num_workers
        self.img_key = [image_key]
        self.label_key = [label_key]
        self.keys = self.img_key + self.label_key
        self.batch_size = self.plan.batch_size

    def setup(self, stage: str):
        def _get_dataset(pimgs, plabels, transforms):
            assert pimgs.exists(), f"the preprocessed img dir {pimgs} do not exists"
            assert (
                plabels.exists()
            ), f"the preprocessed label dir {plabels} do not exists"
            pimgs_files = list(pimgs.iterdir())
            pimgs_files = list(sorted(pimgs_files, key=str))
            plabels_files = list(plabels.iterdir())
            plabels_files = list(sorted(plabels_files, key=str))
            stem = lambda img: filekey(img)
            for pimg, plabel in zip(pimgs_files, plabels_files, strict=True):
                AssertEq(msg="pimg: {}, plabel: {}")(stem(pimg), stem(plabel))

            files = [
                {
                    image_key: pimg,
                    label_key: plabel,
                    "name": stem(pimg),
                }
                for pimg, plabel in zip(pimgs_files, plabels_files, strict=True)
                if pimg.suffix == ".gz" and plabel.suffix == ".gz"
            ]
            return Dataset(files, transforms)

        if stage == "fit":
            pimgs = self.preprocessed_dir / "train" / image_key
            plabels = self.preprocessed_dir / "train" / label_key
            transforms = augmentation_transforms(
                self.plan, self.img_key, self.label_key
            )
            self.train_dataset = _get_dataset(pimgs, plabels, transforms)
            pimgs = self.preprocessed_dir / "val" / image_key
            plabels = self.preprocessed_dir / "val" / label_key
            transforms = Compose([
                load_transformd(self.img_key + self.label_key),
                planned_transformd(self.plan, self.img_key, self.label_key),
            ])
            self.val_dataset = _get_dataset(pimgs, plabels, transforms)

        elif stage == "validate":
            pimgs = self.preprocessed_dir / "val" / image_key
            plabels = self.preprocessed_dir / "val" / label_key
            transforms = Compose([
                load_transformd(self.img_key + self.label_key),
                planned_transformd(self.plan, self.img_key, self.label_key),
            ])
            self.val_dataset = _get_dataset(pimgs, plabels, transforms)
        elif stage == "test":
            pimgs = self.preprocessed_dir / "test" / image_key
            plabels = self.preprocessed_dir / "test" / label_key
            transforms = Compose([
                load_transformd(self.img_key + self.label_key),
                planned_transformd(self.plan, self.img_key, self.label_key),
            ])
            self.test_dataset = _get_dataset(pimgs, plabels, transforms)
        else:
            raise NotImplementedError("Not implemented for " + stage)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1, # 画像サイズを統一できないので1に設定
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
        )
