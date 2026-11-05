# Segmentation validation

`pretrained_seg` and `munet` training validation use the same CPU label
restoration and evaluation as inference (`custom val`, `custom metric`, and
quality observation). Training commands and dataset layout are unchanged.

- Prediction generation is unchanged. `pretrained_seg` uses MONAI sliding-window
  prediction; `munet` blends its patch logits on CPU before argmax.
- Predictions and input metadata move to CPU before inverse transforms. The
  shared `experiments.callbacks.invert` restores labels as inference saving does.
- Validation reads the original input and GT paths retained in MONAI metadata.
  These files must remain readable throughout validation. GT is not reconstructed
  from the cropped/resampled training tensor.
- The shared inference evaluator aligns both labels to the orthogonal grid
  derived from the input image, using nearest-neighbor resampling. This preserves
  the existing inference evaluation grid, including its handling of shear.
- Dice excludes background and ignores case/organ pairs with empty GT (NaN).
  Missed organs with nonempty GT score zero. No surface metric is added.
- `val_dice` is the mean of the non-NaN organ means: first average each organ
  across cases, then average the organs. It corresponds to averaging the Dice
  column of inference's `organ.csv`, excluding NaNs. All-empty evaluation is NaN.
- DDP gathers only scores and deduplicates cases by original input path before
  averaging. NCCL can use CUDA for this small communication; image restoration,
  resampling, and Dice remain on CPU. Scores are cleared at epoch boundaries.

Validation does not write extra prediction or metric files. Existing logging
callbacks retain their previous output behavior. Additional CPU computation,
memory, and GT I/O can increase validation time; forward count does not increase.
Model parameters, checkpoint keys, loss, and inference CSV contents are unchanged.
Old and new `val_dice` logs use different grids/averaging and are not directly
comparable. Checkpoint selection that monitors this value can change.

## Manual verification in the project container

Agents must not execute Python in this repository. Run this check manually:

```bash
uv run --no-sync python -m unittest discover -s tests -p test_segmentation_validation.py
```

The test uses synthetic NIfTI files under a temporary directory, not `data/`.
It checks that CPU validation matches labels saved by the inference logging
callback after crop, spacing, padding, inverse transforms, and grid alignment.
It also checks empty/missed organs, organ macro averaging, reset, and simulated
DDP duplicate removal. It requires the project's imports/dependencies but no
GPU, checkpoint, or medical dataset. Real multi-process DDP is not covered.

For end-to-end verification when compute is available, compare the same model,
plan, cases, and runtime switches in validation and inference. Use eval mode and
disable sampling/augmentation differences. Compare `val_dice` against the mean
of non-NaN organ Dice values in `organ.csv`. Do not compare different subsets
(for example, sanity validation batches versus a full inference set).
