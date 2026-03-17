from abc import ABC, abstractmethod
from pathlib import Path
import argparse
from monai.transforms import Transform
from monai.data import Dataset, DataLoader
import tqdm
from experiments.utils import filekey, resolved_path
import monai.transforms
from experiments.config import image_key, label_key


def load_transformd(keys):
    return monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys),
            monai.transforms.EnsureChannelFirstd(keys),
            monai.transforms.EnsureTyped(keys, data_type="tensor", track_meta=True),
        ]
    )


def planned_transformd(
    plan, image_key: list | tuple = (image_key,), label_key: list | tuple | None = (label_key,)
):
    """planを必要とする処理"""
    if label_key is None:
        all_key = image_key
    else:
        all_key = image_key + label_key
    need_label = label_key is not None
    jplan = plan["configurations"]["3d_fullres"]
    px = jplan["spacing"]
    splan = plan["foreground_intensity_properties_per_channel"]["0"]
    a_min: float = splan[
        "percentile_00_5"
    ]  # 0.5 percentile value of whole training dataset
    a_max: float = splan[
        "percentile_99_5"
    ]  # 99.5 percentile value of whole training dataset
    std: float = splan["std"]  # standard deviation of whole training dataset
    mean: float = splan["mean"]  # average of whole training dataset
    ret = [
        monai.transforms.CropForegroundd(
            all_key,
            select_fn=lambda x: x > -900,
            source_key=image_key[0],
            margin=10,
            allow_smaller=False,
        ),
        monai.transforms.Spacingd(image_key, pixdim=px, mode="bilinear"),
    ]
    if need_label:
        ret.append(monai.transforms.Spacingd(label_key, pixdim=px, mode="nearest"))
    ret.extend(
        [
            monai.transforms.ScaleIntensityRanged(
                image_key,
                a_min=a_min,
                a_max=a_max,
                b_min=a_min,
                b_max=a_max,
                clip=True,
            ),
            monai.transforms.NormalizeIntensityd(
                image_key, subtrahend=mean, divisor=std
            ),
        ]
    )
    return monai.transforms.Compose(ret)


def padded_crop_wrapper(
    keys, crop_size, transforms: list | tuple, padding_factor: float | None = None
):
    """クロップ処理を行った後に回転などが行われて0パディングされるのを防ぐためにあらかじめパディングされたクロップサイズでクロップし, 中の処理が終わってから正しいクロップサイズで切り出すためのラッパー.
    transformsにはアンパック可能なオブジェクトを入れること"""
    if padding_factor is None:
        padding_factor = 1.5
    padded_crop_size = [int(i * padding_factor) for i in crop_size]
    return monai.transforms.Compose(
        [
            monai.transforms.RandSpatialCropd(keys, padded_crop_size),
            *transforms,
            monai.transforms.CenterSpatialCropd(
                keys, crop_size
            ),  # 正しいパッチサイズを切り出す.
        ]
    )


def _load_dir_to_dict(p: Path) -> dict[str, Path]:
    ret = {}
    for f in p.iterdir():
        assert not f.is_dir(), f"a dir {f} found in {p}"
        name = filekey(f)
        ret[name] = f
    return ret


class Preprocessor(ABC):
    def __init__(self, args: list[str]):
        self.parse_args(args)

    def get_argument_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument("-w", "--workers", default=4, type=int)
        return parser

    def parse_args(self, args: list[str]):
        parser = self.get_argument_parser()
        self.args = parser.parse_args(args)

    @abstractmethod
    def preprocess_transform(self) -> Transform: ...

    @abstractmethod
    def initial_data(self) -> list[dict]: ...

    def __call__(self):
        transform = self.preprocess_transform()
        initial_data = self.initial_data()
        dataset = Dataset(initial_data, transform)
        for _ in tqdm.tqdm(
            DataLoader(
                dataset,
                num_workers=self.args.workers,
                batch_size=1,
                collate_fn=lambda x: x[0],
            ),
            dynamic_ncols=True,
        ):
            pass


class ImageOnlyPreprocessor(Preprocessor):
    def get_argument_parser(self) -> argparse.ArgumentParser:
        parser = super().get_argument_parser()
        parser.add_argument("image_path", type=resolved_path)
        parser.add_argument("save_path", type=resolved_path)
        return parser

    def parse_args(self, args: list[str]):
        super().parse_args(args)
        self.image_dir = self.args.image_path
        assert self.image_dir.exists(), f"the raw directory {self.image_dir} not exists"

    def initial_data(self) -> list[dict]:
        rpaird = [
            {image_key: image, "name": name}
            for name, image in _load_dir_to_dict(self.image_dir).items()
        ]
        return rpaird


class ImageLabelPreprocessor(Preprocessor):
    def get_argument_parser(self) -> argparse.ArgumentParser:
        parser = super().get_argument_parser()
        parser.add_argument("image_path", type=resolved_path)
        parser.add_argument("label_path", type=resolved_path)
        parser.add_argument("save_path", type=resolved_path)
        return parser

    def parse_args(self, args: list[str]):
        super().parse_args(args)
        self.image_dir = self.args.image_path
        assert (
            self.image_dir.exists()
        ), f"the raw image directory {self.image_dir} not exists"
        self.label_dir = self.args.label_path
        assert (
            self.label_dir.exists()
        ), f"the raw label directory {self.label_dir} not exists"

    def initial_data(self) -> list[dict]:
        rimgd = _load_dir_to_dict(self.image_dir)
        rlabeld = _load_dir_to_dict(self.label_dir)

        assert set(rlabeld.keys()) == set(
            rimgd.keys()
        ), f"{set(rlabeld.keys()) - set(rimgd.keys())} is only in label dir {self.label_dir}, while {set(rimgd.keys()) - set(rlabeld.keys())} image dir {self.image_dir}"

        rname = set(rimgd.keys())
        rpaird = [
            {image_key: rimgd[name], label_key: rlabeld[name], "name": name}
            for name in rname
        ]
        return rpaird
