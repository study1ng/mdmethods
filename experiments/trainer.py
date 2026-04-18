from pprint import pprint
from lightning import Callback, LightningDataModule, Trainer
from experiments import ArgumentAdaptor
from abc import abstractmethod
from experiments.config import default_training_config, label_key, image_key
from experiments.nets.builder import Builder
from experiments.plan import Plan
from experiments.utils import nowstring, resolved_path
from experiments.assertions import AssertEq
import lightning as L
from pathlib import Path
from experiments.nets.base import UNet
import torch
from torch import tensor, Tensor
from monai.transforms import SaveImaged, SpatialResample
from monai.data import decollate_batch, MetaTensor


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
        print("trainer config: ", config)
        tr = Trainer(**config)
        return tr

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


def _invert_label(label: MetaTensor | Tensor, transform_info):
    cls = transform_info["class"]
    ext = transform_info["extra_info"]
    match cls:
        case "CropForeground":
            if "pad_info" in ext:
                label = _invert_label(label, ext["pad_info"])
            orig = transform_info["orig_size"]
            cropped = ext["cropped"]
            pos = [
                cropped[i] if i % 2 == 0 else orig[i // 2] - cropped[i]
                for i in range(len(cropped))
            ]
            pos = [0, label.shape[0]] + pos
            label_padded = torch.zeros(
                (label.shape[0], *orig), dtype=label.dtype, device=label.device
            )
            indices = tuple(slice(pos[i], pos[i + 1]) for i in range(0, len(pos), 2))
            label_padded[indices] = label
            return label_padded

        case "SpatialPad" | "Pad":
            padded = ext["padded"]
            indices = tuple(
                slice(pad[0], -pad[1] if pad[1] > 0 else None) for pad in padded
            )
            return label[indices]

        case "SpatialResample":
            a = ext["src_affine"]
            ia = torch.inverse(a)
            resampler = SpatialResample(
                mode=ext["mode"],
                align_corners=ext["align_corners"],
                padding_mode=ext["padding_mode"],
            )
            output_data = resampler(
                img=label,
                dst_affine=ia,
                spatial_size=transform_info["orig_size"],
            )
            return output_data

        case _:
            raise NotImplementedError(f"{cls} is not implemented")


def invert_label(
    item: dict[str, MetaTensor | Tensor], image_key=image_key, label_key=label_key
) -> MetaTensor:
    image = item[image_key]
    label = item[label_key]
    transforms = image.applied_operations
    for transform_info in reversed(transforms):
        try:
            label = _invert_label(label, transform_info)
        except Exception as e:
            print(label.shape)
            pprint(transform_info)
            raise e from e
    item[label_key] = MetaTensor(label, meta=item[image_key].meta).to(torch.int16)
    return item


class InferenceCallback(Callback):
    def __init__(
        self,
        save_path,
        label_key=label_key,
        image_key=image_key,
    ):
        super().__init__()
        self.save_path = save_path
        self.label_key = label_key
        self.image_key = image_key
        self.saver = SaveImaged(
            output_dir=self.save_path, output_postfix="", keys=self.label_key
        )
        # Invertdはidで引っかかるので手動で逆変換

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        out = torch.argmax(outputs, dim=1, keepdim=True)
        batch[self.label_key] = out
        for item in decollate_batch(batch):
            item = invert_label(
                item, image_key=self.image_key, label_key=self.label_key
            )
            self.saver(item)


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
            callbacks=[InferenceCallback(self.save_path)], devices=self.devices
        )
        return tr

    def parse_args(self, args):
        super().parse_args(args)
        self.plan = Plan(self.args.plan_path)
