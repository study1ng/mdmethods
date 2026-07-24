from abc import ABC, abstractmethod
from random import randint
import lightning as L
from pathlib import Path
from monai.data import DataLoader, Dataset
import monai.transforms
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
    SpatialPadd,
    Identityd,
)
import torch
from pathlib import Path
import numpy as np
from experiments.config import image_key, label_key, filekey
from experiments.preprocess import (
    load_transformd,
    planned_transformd,
    padded_crop_wrapper,
)
from experiments.plan import Plan


class UNetDataModule(ABC, L.LightningDataModule):
    def __init__(
        self,
        plan: Plan,
        num_workers: int = 4,
    ):
        super().__init__()
        self.plan = plan
        self.num_workers = num_workers
        self.batch_size = self.plan.batch_size

    @abstractmethod
    def _get_dataset(self, *, stage, transforms): ...

    @abstractmethod
    def get_fit_transforms(self): ...

    @abstractmethod
    def get_val_transforms(self): ...

    @abstractmethod
    def get_test_transforms(self): ...

    def setup(self, stage: str):
        if stage == "fit":
            transforms = self.get_fit_transforms()
            self.train_dataset = self._get_dataset(stage=stage, transforms=transforms)
            self.setup("validate")
        elif stage == "validate":
            transforms = self.get_val_transforms()
            self.val_dataset = self._get_dataset(stage=stage, transforms=transforms)
        elif stage == "test":
            transforms = self.get_test_transforms()
            self.test_dataset = self._get_dataset(stage=stage, transforms=transforms)
        else:
            raise NotImplementedError(f"{stage} is not implemented")

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
            batch_size=1,
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


class FsDataModule(UNetDataModule):
    def __init__(
        self,
        data_locator: str | Path,
        plan: Plan,
        num_workers: int = 4,
    ):
        super().__init__(plan=plan, num_workers=num_workers)
        self.data_locator = Path(data_locator)

    def _get_dataset(self, *, stage, transforms):
        if stage == "fit":
            pimgs = self.data_locator / "train" / image_key
            plabels = self.data_locator / "train" / label_key
        elif stage == "validate":
            pimgs = self.data_locator / "val" / image_key
            plabels = self.data_locator / "val" / label_key
        elif stage == "test":
            pimgs = self.data_locator / "test" / image_key
            plabels = None
        else:
            raise NotImplementedError(f"{stage} is not implemented")

        assert pimgs.exists(), f"the preprocessed img dir {pimgs} do not exists"
        pimgs_files = list(pimgs.iterdir())
        pimgs_files = list(sorted(pimgs_files, key=str))
        if plabels is not None:
            assert (
                plabels.exists()
            ), f"the preprocessed label dir {plabels} do not exists"
            plabels_files = list(plabels.iterdir())
            plabels_files = list(sorted(plabels_files, key=str))
            for pimg, plabel in zip(pimgs_files, plabels_files, strict=True):
                assert filekey(pimg) == filekey(
                    plabel
                ), f"pimg: {pimg}, plabel: {plabel}"

            files = [
                {
                    image_key: pimg,
                    label_key: plabel,
                    "name": filekey(pimg),
                }
                for pimg, plabel in zip(pimgs_files, plabels_files, strict=True)
                if pimg.suffix == ".gz" and plabel.suffix == ".gz"
            ]
            return Dataset(files, transforms)
        else:
            return Dataset(
                [
                    {image_key: pimg, "name": filekey(pimg)}
                    for pimg in pimgs_files
                    if pimg.suffix == ".gz"
                ],
                transforms,
            )


class NoSpecialDataModule(UNetDataModule):
    def get_val_transforms(self):
        all_key = [image_key] + [label_key] if label_key is not None else image_key
        composelist = [
            load_transformd(all_key),
            planned_transformd(self.plan, image_key, label_key),
        ]
        return Compose(composelist)

    def get_test_transforms(self):
        composelist = [
            load_transformd(image_key),
            planned_transformd(self.plan, image_key),
        ]
        return Compose(composelist)


class CropDataModule(NoSpecialDataModule):
    def get_fit_transforms(self):
        plan = self.plan
        all_key = [image_key] + [label_key] if label_key is not None else image_key
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
                image_key,
                prob=0.1,
                gamma=(0.7, 1.5),
                invert_image=True,
                retain_stats=True,
            ),
            RandSimulateLowResolutiond(
                image_key,
                prob=0.25,
                downsample_mode="nearest",
                upsample_mode="trilinear",
                zoom_range=(0.5, 1.0),
            ),
        ]
        composelist += [
            RandFlipd(all_key, prob=0.5, spatial_axis=0),
            RandFlipd(all_key, prob=0.5, spatial_axis=1),
            RandFlipd(all_key, prob=0.5, spatial_axis=2),
        ]
        return Compose(composelist)


class NoCropDataModule(NoSpecialDataModule):
    def get_fit_transforms(
        self,
    ):
        plan = self.plan
        global image_key, label_key
        image_key = [image_key]
        label_key = [label_key]
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
            RandZoomd(
                all_key,
                prob=0.2,
                min_zoom=min_zoom,
                max_zoom=max_zoom,
                mode=interp_modes,
                keep_size=True,
            ),
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
                image_key,
                prob=0.1,
                gamma=(0.7, 1.5),
                invert_image=True,
                retain_stats=True,
            ),
            RandSimulateLowResolutiond(
                image_key,
                prob=0.25,
                downsample_mode="nearest",
                upsample_mode="trilinear",
                zoom_range=(0.5, 1.0),
            ),
        ]
        composelist += [
            RandFlipd(all_key, prob=0.5, spatial_axis=0),
            RandFlipd(all_key, prob=0.5, spatial_axis=1),
            RandFlipd(all_key, prob=0.5, spatial_axis=2),
            SpatialPadd(keys=all_key, spatial_size=patch_size),
        ]
        return Compose(composelist)


class DummyDataModule(UNetDataModule):
    """DataModule raises Dummy data for profiling"""

    def __init__(
        self,
        plan,
        num_workers=4,
        *,
        sorter=lambda x: sorted(x, key=lambda f: f[0].numel()),
        dim: int,
        num_samples: int = 30,
        lower: int,
        upper: int,
    ):
        self.sorter = sorter
        self.dim = dim
        self.num_samples = num_samples
        self.lower = lower
        self.upper = upper
        super().__init__(plan, num_workers)

    def produce(self, num_size: int, stage, **kwargs):
        def _shape():
            c = 1
            s = tuple(randint(self.lower, self.upper) for _ in range(self.dim))
            return (c, *s)
        shapes = [_shape() for _ in range(self.num_samples)]
        return tuple(
            tuple(torch.rand(s, **kwargs) for _ in range(num_size))
            for s in shapes
        )

    def _get_dataset(self, *, stage, transforms):
        files = []
        for img, label in self.sorter(self.produce(2, stage)):
            img = img.float()
            label = label.int()
            files.append(
                {
                    image_key: img,
                    label_key: label,
                    "name": "dummy"
                }
            )

        return Dataset(files, transform=transforms)
    def get_fit_transforms(self):
        all_key = [image_key] + [label_key] if label_key is not None else image_key
        return Identityd(all_key)

    def get_val_transforms(self):
        all_key = [image_key] + [label_key] if label_key is not None else image_key
        return Identityd(all_key)

    def get_test_transforms(self):
        all_key = [image_key] + [label_key] if label_key is not None else image_key
        return Identityd(all_key)

class CropFsDataModule(FsDataModule, CropDataModule):
    pass


class NoCropFsDataModule(FsDataModule, NoCropDataModule):
    pass
