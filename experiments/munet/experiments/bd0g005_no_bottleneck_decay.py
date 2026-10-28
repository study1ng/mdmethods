"""bd0 with initial gamma=0.05 and no decay on the added bottleneck.

Select with --experiment_name bd0g005_no_bottleneck_decay for train, inference,
or custom val. Training uses the same arguments and data layout as bd0:

    uv run python main.py munet train --experiment_name bd0g005_no_bottleneck_decay ...

Start a new run from the original pretrained UNet via --pretrained_path.
The encoder stays frozen, overlap_scale is 0, and decoder decay remains 0.1
except for Mamba A_log/D. All added bottleneck parameters, including gamma,
use zero decay. Optimizer parameter grouping has changed: old model weights
remain loadable, but old optimizer states cannot be resumed unchanged.
"""

from experiments.munet import inference, custom
from experiments.munet.experiments.bd0 import BottleneckSeg as BaseBottleneckSeg
from experiments.munet.model import MUNetTrainingModule as Model
from experiments.nets.builder import Builder
from experiments.pretrained_seg import analyze, prune


class BottleneckSeg(BaseBottleneckSeg):
    def _build_module(self):
        if self.args.pretrained_path is None:
            raise Exception("Munet needs pretrained model")
        builder = Builder().based_on_ckpt(self.args.pretrained_path).to_params()
        return Model(
            builder=builder,
            plan=self.plan,
            overlap_scale=0.0,
            gamma=0.05,
            bottleneck_weight_decay=0.0,
        )


def train(args, parsed):
    BottleneckSeg(args, parsed)()
