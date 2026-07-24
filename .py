import nibabel as nib
import numpy as np
import torch

from monai.metrics import DiceMetric


def calculate_val_dice_like_validation(gt_path, pred_path):
    gt = nib.load(gt_path).get_fdata().astype(np.int64)
    pred = nib.load(pred_path).get_fdata().astype(np.int64)

    if gt.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch: GT {gt.shape}, Pred {pred.shape}"
        )

    num_classes = int(max(gt.max(), pred.max()) + 1)

    # validation_step と同じ形式 [B, 1, H, W, D]
    gt = torch.from_numpy(gt).long().unsqueeze(0).unsqueeze(0)
    pred = torch.from_numpy(pred).long().unsqueeze(0).unsqueeze(0)

    metric = DiceMetric(
        include_background=False,
        reduction="mean",
        ignore_empty=True,
        num_classes=num_classes,
    )

    metric(pred, gt)

    dice = metric.aggregate().item()

    metric.reset()

    return dice


gt_path = "./data/trained/ttsg/pretrained_seg/2026.07.21.01.54.08/99_1138_gt.nii.gz"
pred_path = "./data/trained/ttsg/pretrained_seg/2026.07.21.01.54.08/99_1138_out.nii.gz"

dice = calculate_val_dice_like_validation(gt_path, pred_path)

print(f"Dice = {dice:.6f}")