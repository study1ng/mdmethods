"""CPU validation using the same restoration and evaluator as inference."""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import MetaTensor, decollate_batch

from experiments.callbacks import invert
from experiments.config import image_key, label_key


class SegmentationValidation:
    """Keep only per-case scores; never retain epoch-sized image tensors."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cases = {}

    @torch.no_grad()
    def update(self, prediction, batch, num_classes):
        # Import at runtime: munet's package imports pretrained_seg's model.
        from experiments.munet.evaluation import _evaluate_case

        images = decollate_batch(batch[image_key].detach().cpu())
        labels = decollate_batch(batch[label_key].detach().cpu())
        predictions = decollate_batch(prediction.detach().cpu())
        for image, label, predicted in zip(images, labels, predictions, strict=True):
            if not isinstance(image, MetaTensor) or not isinstance(label, MetaTensor):
                raise TypeError("Validation requires image and label metadata")
            source = {
                "input_image": Path(image.meta["filename_or_obj"]),
                "reference_label": Path(label.meta["filename_or_obj"]),
            }
            # Match LogCallback: labels inherit the input image's transform history.
            restored = invert(
                MetaTensor(predicted, meta=dict(image.meta)), image, is_label=True,
            )
            prediction_image = nib.Nifti1Image(
                restored.as_tensor()[0].numpy().astype(np.int64),
                restored.affine.numpy(), dtype=np.int64,
            )
            prediction_image.header.set_xyzt_units("mm")
            # Match saved/reloaded NIfTI affine precision, as observe does.
            prediction_image = nib.Nifti1Image(
                prediction_image.dataobj, prediction_image.header.get_best_affine(),
                header=prediction_image.header,
            )
            name = str(source["input_image"].resolve())
            self.cases[name] = _evaluate_case(
                name, source, prediction_image, num_classes,
                range(1, num_classes), True, None, None,
            )

    def aggregate(self):
        from experiments.munet.evaluation import mean_metric_rows, _mean

        cases = self.cases
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            # Scores only. NCCL may use CUDA internally for this small collective.
            torch.distributed.all_gather_object(gathered, cases)
            cases = {}
            for rank_cases in gathered:
                for name, rows in rank_cases.items():
                    # DistributedSampler pads with repeated cases; count each once.
                    cases.setdefault(name, rows)
        rows = [row for name in sorted(cases) for row in cases[name]]
        organs = mean_metric_rows(rows, "organ", ["dice"])
        return _mean([row["dice"] for row in organs])
