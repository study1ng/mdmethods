from abc import ABC, abstractmethod
import lightning as L
from experiments.plan import Plan
from pathlib import Path
from experiments.config import image_key, label_key, filekey
from monai.data import DataLoader, Dataset

class UNetDataModule(ABC, L.LightningDataModule):
    def __init__(
        self,
        data_locator: str | Path,
        plan: Plan,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_locator = Path(data_locator)
        self.plan = plan
        self.num_workers = num_workers
        self.img_key = [image_key]
        self.label_key = [label_key]
        self.keys = self.img_key + self.label_key
        self.batch_size = self.plan.batch_size

    def _get_dataset(self, pimgs, plabels=None, *, transforms):
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
                assert filekey(pimg) == filekey(plabel), f"pimg: {pimg}, plabel: {plabel}"
                
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
                [{image_key: pimg, "name": filekey(pimg)}
                for pimg in pimgs_files
                if pimg.suffix == ".gz"],
                transforms
            )
        
    def setup(self, stage: str):
        if stage == "fit":
            pimgs = self.data_locator / "train" / image_key
            plabels = self.data_locator / "train" / label_key
            transforms = augmentation_transforms(
                self.plan, self.img_key, self.label_key
            )
            self.train_dataset = self._get_dataset(pimgs, plabels, transforms=transforms)
            pimgs = self.data_locator / "val" / image_key
            plabels = self.data_locator / "val" / label_key
            transforms = val_transforms(self.plan, self.img_key, self.label_key)
            self.val_dataset = self._get_dataset(pimgs, plabels, transforms=transforms)

        elif stage == "validate":
            pimgs = self.data_locator / "val" / image_key
            plabels = self.data_locator / "val" / label_key
            transforms = val_transforms(self.plan, self.img_key, self.label_key)
            self.val_dataset = self._get_dataset(pimgs, plabels, transforms=transforms)
        elif stage == "test":
            pimgs = self.data_locator / "test" / image_key
            transforms = test_transforms(self.plan, self.img_key)
            self.test_transforms = transforms
            self.test_dataset = self._get_dataset(pimgs, transforms=transforms)
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
            batch_size=1,  # 画像サイズを統一できないので1に設定
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


    @abstractmethod
    def need_prune(self, data):
        ...