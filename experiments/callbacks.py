import lightning as L
from copy import deepcopy
from lightning.pytorch.callbacks import Callback
import nibabel
from torch import Tensor
from torch.linalg import inv
from monai.data import decollate_batch, MetaTensor
import torch
from monai.transforms import SpatialResample, SaveImage, SaveImaged, Rotate, Zoom
from monai import transforms
from pprint import pprint
from experiments.config import label_key, image_key
from experiments.utils.wraputils import element_wise
from lightning.pytorch.trainer.states import RunningStage

def _invert(
    item: MetaTensor | Tensor,
    transform_info,
    *,
    resample_mode: str | None = None,
):
    cls = transform_info["class"]
    ext = transform_info["extra_info"]
    match cls:
        case "CropForeground":
            if "pad_info" in ext:
                item = _invert(
                    item,
                    ext["pad_info"],
                    resample_mode=resample_mode,
                )
            orig = transform_info["orig_size"]
            cropped = ext["cropped"]
            pos = [
                cropped[i] if i % 2 == 0 else orig[i // 2] - cropped[i]
                for i in range(len(cropped))
            ]
            pos = [0, item.shape[0]] + pos
            label_padded = torch.zeros(
                (item.shape[0], *orig), dtype=item.dtype, device=item.device
            )
            indices = tuple(slice(pos[i], pos[i + 1]) for i in range(0, len(pos), 2))
            label_padded[indices] = item
            return label_padded

        case "SpatialPad" | "Pad":
            padded = ext["padded"]
            indices = tuple(
                slice(pad[0], -pad[1] if pad[1] > 0 else None) for pad in padded
            )
            return item[indices]

        case "SpatialResample":
            resampler = SpatialResample(
                mode=resample_mode or ext["mode"],
                align_corners=ext["align_corners"],
                padding_mode=ext["padding_mode"],
            )
            output_data = resampler(
                img=item,
                dst_affine=ext["src_affine"],
                spatial_size=transform_info["orig_size"],
            )
            return output_data
        
        case "CenterSpatialCrop" | "RandSpatialCrop":
            cropped = ext["cropped"]
            orig = transform_info["orig_size"]
            pos = [
                cropped[i] if i % 2 == 0 else orig[i // 2] - cropped[i]
                for i in range(len(cropped))
            ]
            pos = [0, item.shape[0]] + pos
            label_padded = torch.zeros(
                (item.shape[0], *orig), dtype=item.dtype, device=item.device
            )
            indices = tuple(slice(pos[i], pos[i + 1]) for i in range(0, len(pos), 2))
            label_padded[indices] = item
            return label_padded
        
        case "RandRotated" | "RandFlipd" | "RandZoomd" | "RandZoom" | "RandRotate":
            if "class" in ext:
                return _invert(item, ext, resample_mode=resample_mode)
            return item
        
        case "Flip":
            axes = ext["axes"]
            axes = (axes,) if isinstance(axes, int) else axes
            return torch.flip(item, axes)
            
        case "Zoom":
            return Zoom.inverse_transform(None, item, transform_info)
        
        case "Rotate":
            return Rotate.inverse_transform(None, item, transform_info)

        case _:
            raise NotImplementedError(f"{cls} is not implemented")


def invert(
    value: MetaTensor | Tensor,
    reference: MetaTensor,
    *,
    is_label: bool,
) -> MetaTensor:
    transforms = reference.applied_operations
    for transform_info in reversed(transforms):
        try:
            transform_info = deepcopy(transform_info)
            value = _invert(
                value,
                transform_info,
                resample_mode="nearest" if is_label else None,
            )
        except Exception as e:
            print(value.shape)
            pprint(transform_info)
            raise e from e

    meta = dict(reference.meta)
    original_affine = meta.get("original_affine", reference.affine)
    original_shape = meta.get("spatial_shape")
    meta.pop("affine", None)
    if original_shape is not None:
        original_shape = tuple(int(size) for size in original_shape)
        if tuple(value.shape[1:]) != original_shape:
            raise RuntimeError(
                "inverse transform did not restore the original spatial shape: "
                f"expected {original_shape}, got {tuple(value.shape[1:])}"
            )
        meta["spatial_shape"] = original_shape
    value = MetaTensor(value, affine=original_affine, meta=meta)
    return value.to(torch.int16) if is_label else value


class LogCallback(Callback):
    def __init__(
        self,
        save_path,
        *,
        label_key=label_key,
        image_key=image_key,
        on_train_end = True,
        on_val_end = True,
        on_test_end = True,
    ):
        super().__init__()
        self.save_path = save_path
        self.label_key = label_key
        self.image_key = image_key
        self.on_train_end = on_train_end
        self.on_val_end = on_val_end
        self.on_test_end = on_test_end


    def _print_summary(self, k, v):
        """t == 'summary' の際に統計指標を計算して出力するヘルパーメソッド"""
        if not isinstance(v, torch.Tensor):
            print(f"[{k}] Summary: Value is not a torch.Tensor")
            return
        
        v_float = v.float()
        
        if v_float.numel() == 0:
            print(f"[{k}] Summary: Empty tensor")
            return

        min_val = v_float.min().item()
        max_val = v_float.max().item()
        mean_val = v_float.mean().item()
        median_val = v_float.median().item()
        
        std_val = v_float.std().item() if v_float.numel() > 1 else 0.0
        
        q25 = torch.quantile(v_float, 0.25).item()
        q75 = torch.quantile(v_float, 0.75).item()
        sparsity = (v_float == 0).sum().item() / v_float.numel()

        print(f"\n--- Summary for [{k}] ---")
        print(f"Min: {min_val:.4f} | Max: {max_val:.4f} | Mean: {mean_val:.4f}")
        print(f"Median: {median_val:.4f} | Std: {std_val:.4f}")
        print(f"25th %ile: {q25:.4f} | 75th %ile: {q75:.4f} | Zeros: {sparsity:.2%}")
        print("-" * 25 + "\n")

    def _process_action(self, stage, k, batch, epoch, action: tuple[str | tuple[str, ...], MetaTensor], test: bool):
        action, v = action

        @element_wise(types=str)
        def _process(action: str):
            nonlocal v
            if action == "summary":
                self._print_summary(k, v)
                return
            if action == "label":
                v = torch.argmax(v, dim=1, keepdim=True)
            bk = self.label_key if action == "label" else self.image_key
            items = decollate_batch(batch)
            values = decollate_batch(v)
            for item, value in zip(items, values, strict=True):
                image_meta = (
                    item[self.image_key].meta
                    if isinstance(item[self.image_key], MetaTensor)
                    else {}
                )
                reference = item[self.image_key]
                is_label = action == "label" or k in {"label", "gt", "out"}
                item[bk] = MetaTensor(value, meta=dict(image_meta))
                if isinstance(reference, MetaTensor):
                    item[bk] = invert(
                        item[bk],
                        reference,
                        is_label=is_label,
                    )
                origstem = item.get("name")
                if not isinstance(origstem, str):
                    orig = image_meta.get("filename_or_obj", "unknown.nii.gz")
                    origstem = (
                        orig.split("/")[-1].split(".")[0]
                        if isinstance(orig, str)
                        else "unknown"
                    )
                item[bk].meta["filename_or_obj"] = f"{stage}_{epoch}_{origstem}_{k}.nii.gz"
                if item[bk].dtype == torch.bfloat16:
                    item[bk] = item[bk].to(torch.float32)
                SaveImage(output_dir=self.save_path, output_postfix="", separate_folder=False)(item[bk])
        
        _process(action)

    def process_action(self, trainer: L.Trainer, batch, outputs):
        if trainer.state.stage == RunningStage.TRAINING:
            stage = "fit"
        elif trainer.state.stage == RunningStage.VALIDATING:
            stage = "val"
        elif trainer.state.stage == RunningStage.TESTING:
            stage = "test"
        elif trainer.state.stage == RunningStage.PREDICTING:
            stage = "predict"
        else:
            stage = "unknown"
        for k, action in outputs.items():
            if not isinstance(action, tuple):
                continue
            self._process_action(stage, k, batch, trainer.current_epoch, action, trainer.testing)


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.on_train_end:
            return
        if batch_idx + 1 != trainer.num_training_batches:
            return
        self.process_action(trainer, batch, outputs)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        if not self.on_val_end:
            return
        num_batches = trainer.num_val_batches
        if isinstance(num_batches, list):
            num_batches = num_batches[dataloader_idx]
        if batch_idx + 1 != num_batches:
            return
        self.process_action(trainer, batch, outputs)

    def on_test_batch_end(
        self, trainer, pl_module, outputs: dict[str, tuple[str, MetaTensor]], batch, batch_idx, dataloader_idx=0
    ):
        if not self.on_test_end:
            return
        # これは推論/保存も兼ねているのですべてのパッチに対して行う.
        self.process_action(trainer, batch, outputs)


class OOMExceptionCallback(Callback):
    def __init__(self):
        super().__init__()
        torch.cuda.memory._record_memory_history("state", )
    
    def dump_cuda_tensors(self):
        import gc
        import torch

        tensors = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    size = obj.numel() * obj.element_size() / 1024**2
                    if size > 100:
                        print(
                            f"{size:.1f}MB "
                            f"{obj.shape} "
                            f"ptr={obj.data_ptr()}"
                        )
            except:
                pass

        tensors.sort(key=lambda x: x.numel() * x.element_size(), reverse=True)

        for t in tensors[:20]:
            size_mb = t.numel() * t.element_size() / 1024**2
            print(
                f"{size_mb:8.1f} MB "
                f"shape={tuple(t.shape)} "
                f"dtype={t.dtype} "
                f"grad={t.requires_grad}"
            )
    
    def on_exception(self, trainer, pl_module, exception):
        torch.cuda.memory._dump_snapshot("mem_snapshot.html")
        print(torch.cuda.memory_summary())
        self.dump_cuda_tensors()    
