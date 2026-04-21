from experiments.nets.builder import Builder
from experiments.pretrained_seg import PlainSegmentation, analyze, prune, inference
from experiments.pretrained_seg.model import SegmentationModule as Model

# TODO: Finetuning need a callback which freeze its encoder weight at first and unfreeze from 300th epochs

class Finetuning(PlainSegmentation):
    def _build_module(self):
        builder = Builder()
        if self.args.pretrained_path is not None:
            builder = builder.based_on_ckpt(self.args.pretrained_path).reinitialize(
                "nets.plainunet.PlainHead"
            )
        else:
            raise Exception("Lora needs pretrained model")
        builder.lora(
            target_module_type=[
                "torch.nn.Conv3d",
                "torch.nn.Linear",
                "torch.nn.Conv1d",
                "torch.nn.Conv2d",
            ],
            skips=["decoder.head"],
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["decoder.head"],
        )
        builder = builder.to_params()
        lm = Model(builder=builder, plan=self.plan)
        return lm

def train(args, parsed):
    Finetuning(args, parsed)()
