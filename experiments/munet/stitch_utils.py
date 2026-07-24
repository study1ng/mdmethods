from __future__ import annotations
import torch
from monai.utils import (
    BlendMode,
)
from monai.data.utils import iter_patch_position
from experiments.utils import get_gaussian_kernel
import numpy as np


def split_to_patch(
    image: torch.Tensor, *, patch_size, overlap
) -> tuple[tuple[torch.Tensor, torch.Tensor | None, tuple[int, ...]], ...]:
    """
    Returns:
        ((patch_image, patch_label, patch_position)*)
    """
    # -> [(patch_image, patch_label, patch_position)*]
    # patch_positionはパッチの開始位置(バッチ次元, チャネル次元は含まない)
    patch_start_gen = iter_patch_position(
        image.shape[2:], patch_size, overlap=overlap
    )
    patch_slice_iter = (
        tuple(
            slice(start, start + patch_len)
            for start, patch_len in zip(patch_start, patch_size)
        )
        for patch_start in patch_start_gen
    )
    patches = tuple(
        (
            image[(...,) + slice_tuple],
            tuple(s.start for s in slice_tuple),
        )
        for slice_tuple in patch_slice_iter
    )
    return patches

def stitch_logits(
    patches: tuple[tuple[torch.Tensor, tuple[int, ...]], ...],
    mode=BlendMode.CONSTANT,
    sigma_scale=0.125,
    *,
    output_size: tuple[int, ...] | None = None,
):
    try:
        first_patch, _ = patches[0]

        # 元のデバイスを保持
        original_device = first_patch.device

        # CPUへ移動
        first_patch_cpu = first_patch.cpu()

        patch_shape = np.array(first_patch_cpu.shape[2:])

        importance_map = (
            torch.ones_like(first_patch_cpu)
            if mode == BlendMode.CONSTANT
            else get_gaussian_kernel(
                tuple(patch_shape), sigma_scale
            ).to("cpu")
        )

        if output_size is None:
            output_size = np.zeros(len(patch_shape), dtype=int)

            for patch, patch_pos in patches:
                patch_pos = np.array(patch_pos)
                patch_end = patch_pos + np.array(patch.shape[2:])
                output_size = np.maximum(output_size, patch_end)

        canvas = torch.zeros(
            (*first_patch_cpu.shape[:2], *output_size),
            dtype=first_patch_cpu.dtype,
            device="cpu",
        )

        weight_map = torch.zeros_like(canvas)

        for patch, patch_pos in patches:
            patch_cpu = patch.cpu()

            patch_pos = np.array(patch_pos)
            patch_end = patch_pos + np.array(patch_cpu.shape[2:])

            slc = (
                slice(None),
                slice(None),
                *[slice(s, e) for s, e in zip(patch_pos, patch_end)],
            )

            canvas[slc].addcmul_(patch_cpu, importance_map)
            weight_map[slc] += importance_map

        if mode != BlendMode.CONSTANT:
            weight_map.clamp_(min=1e-8)
            canvas.div_(weight_map)

        # 元のデバイスへ戻す
        return canvas.to(original_device)

    except torch.OutOfMemoryError as e:
        print(patches[0][0].shape, len(patches))
        raise e