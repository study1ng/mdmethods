"""maskratio0_6 training configuration with patch encoder feature inference."""

from experiments.mim.experiments.maskratio0_6 import MIM, analyze, prune, train
from experiments.encoder_features import EncoderFeatureInferencer


def inference(args, meta):
    EncoderFeatureInferencer(args, meta)()

