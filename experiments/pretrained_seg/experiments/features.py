"""pretrained_seg training configuration with patch encoder feature inference."""

from experiments.pretrained_seg import PlainSegmentation, analyze, prune, train
from experiments.encoder_features import EncoderFeatureInferencer


def inference(args, meta):
    EncoderFeatureInferencer(args, meta)()

