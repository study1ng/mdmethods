from experiments.config import default_training_config
from experiments.pretrained_seg import (
    prune,
    analyze,
    preprocess,
    PlainSegmentation,
    PlainSegInferencer
)
from experiments.nets.ubimamba import UBiMamba as UNet
from experiments.pretrained_seg.model import SegmentationModule as Model
from lightning.pytorch.callbacks import BaseFinetuning
from lightning import Trainer
import torch
from experiments.utils.fsutils import resolved_path


class ZeroshotFinetuning(BaseFinetuning):
    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.unet.encoder)
        self.freeze(pl_module.unet.decoder.stages)
        pl_module.deep_supervision = False
        pl_module.unet.deep_supervision = False
        pl_module.unet.decoder.deep_supervision = False # ゼロショットなので深層監督は不要

    def finetune_function(self, pl_module, epoch, optimizer):
        pass

class ZeroshotMaskfeat(PlainSegmentation):
    def get_argument_parser(self):
        parser = super().get_argument_parser()
        parser.add_argument("pretrained_path", type=resolved_path, default=None)
        return parser
    def _build_trainer(self):
        config = default_training_config(
            save_path=self.save_path, meta=self.meta, devices=self.devices
        )
        config["max_epochs"] = 100
        config["min_epochs"] = 100
        config["callbacks"].append(ZeroshotFinetuning())
        print("trainer config: ", config)
        tr = Trainer(**config)
        return tr

    def _build_module(self):
        unet = UNet.from_plan(self.plan, 1, deep_supervision=True, output_channel=118)
        lm = Model(unet=unet, plan=self.plan)
        return lm


def train(args, parsed):
    ZeroshotMaskfeat(args, parsed)()

class Inference(PlainSegInferencer):
    def _build_module(self):
        unet = UNet.from_plan(self.plan, 1, deep_supervision=True, output_channel=118)
        checkpoint = torch.load(self.ckpt_path, map_location=lambda storage, loc: storage)
        if "unet_hparams" in checkpoint["hyper_parameters"]:
            del checkpoint["hyper_parameters"]["unet_hparams"]
        model = Model(**checkpoint["hyper_parameters"], unet=unet)
        model.load_state_dict(checkpoint["state_dict"])
        return model

def inference(args, parsed):
    Inference(args, parsed)()