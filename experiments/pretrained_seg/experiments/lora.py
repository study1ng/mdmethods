from experiments.nets.builder import Builder
from experiments.pretrained_seg import PlainSegmentation, analyze, prune, inference
from experiments.pretrained_seg.model import SegmentationModule as Model
from experiments.config import default_training_config
from lightning import Trainer


class LoRA(PlainSegmentation):
    def _build_module(self):
        builder = Builder()
        if self.args.pretrained_path is not None:
            builder = builder.based_on_ckpt(self.args.pretrained_path).reinitialize(
                "nets.plainunet.PlainHead",
                output_channel = 118,
            )
        else:
            raise Exception("Lora needs pretrained model")
        builder.lora(
            target_module_type=[
                "^(encoder\\.stages\\.[3456789]|decoder\\.stages).*/torch.nn.Conv3d/torch.nn.Linear",
            ],
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["^decoder\\.head.*/torch.nn.Conv3d/torch.nn.Linear"]
        )
        builder = builder.to_params()
        lm = Model(builder=builder, plan=self.plan)
        return lm
    
    def _build_trainer(self):
        config = default_training_config(
            save_path=self.save_path, meta=self.meta, devices=self.devices
        )
        config["max_epochs"] = 100
        config["min_epochs"] = 100
        print("trainer config: ", config)
        tr = Trainer(**config)
        return tr
    
def train(args, parsed):
    LoRA(args, parsed)()
