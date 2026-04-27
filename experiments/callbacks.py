import lightning as L
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

def _invert_label(label: MetaTensor | Tensor, transform_info):
    if not isinstance(label, MetaTensor):
        label = MetaTensor(label)
    cls = transform_info["class"]
    ext = transform_info["extra_info"]
    match cls:
        case "CropForeground":
            if "pad_info" in ext:
                label = _invert_label(label, ext["pad_info"])
            orig = transform_info["orig_size"]
            cropped = ext["cropped"]
            pos = [
                cropped[i] if i % 2 == 0 else orig[i // 2] - cropped[i]
                for i in range(len(cropped))
            ]
            pos = [0, label.shape[0]] + pos
            label_padded = torch.zeros(
                (label.shape[0], *orig), dtype=label.dtype, device=label.device
            )
            indices = tuple(slice(pos[i], pos[i + 1]) for i in range(0, len(pos), 2))
            label_padded[indices] = label
            return label_padded

        case "SpatialPad" | "Pad":
            padded = ext["padded"]
            indices = tuple(
                slice(pad[0], -pad[1] if pad[1] > 0 else None) for pad in padded
            )
            return label[indices]

        case "SpatialResample":
            a = ext["src_affine"]
            ia = torch.inverse(a)
            resampler = SpatialResample(
                mode=ext["mode"],
                align_corners=ext["align_corners"],
                padding_mode=ext["padding_mode"],
            )
            output_data = resampler(
                img=label,
                dst_affine=ia,
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
            pos = [0, label.shape[0]] + pos
            label_padded = torch.zeros(
                (label.shape[0], *orig), dtype=label.dtype, device=label.device
            )
            indices = tuple(slice(pos[i], pos[i + 1]) for i in range(0, len(pos), 2))
            label_padded[indices] = label
            return label_padded
        
        case "RandRotated" | "RandFlipd" | "RandZoomd" | "RandZoom" | "RandRotate":
            if "class" in ext:
                return _invert_label(label, ext)
            return label
        
        case "Flip":
            axes = ext["axes"]
            axes = (axes,) if isinstance(axes, int) else axes
            return torch.flip(label, axes)
            
        case "Zoom":
            return Zoom.inverse_transform(None, label, transform_info)
        
        case "Rotate":
            return Rotate.inverse_transform(None, label, transform_info)

        case _:
            raise NotImplementedError(f"{cls} is not implemented")


def invert_label(
    item: dict[str, MetaTensor | Tensor], image_key=image_key, label_key=label_key
) -> MetaTensor:
    image = item[image_key]
    label = item[label_key]
    transforms = image.applied_operations
    for transform_info in reversed(transforms):
        try:
            label = _invert_label(label, transform_info)
        except Exception as e:
            print(label.shape)
            pprint(transform_info)
            raise e from e
    item[label_key] = MetaTensor(label, meta=item[image_key].meta).to(torch.int16)
    return item


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

    def _process_action(self, k, batch, epoch, action: tuple[str | tuple[str, ...], MetaTensor]):
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
            batch[bk] = v
            for item in decollate_batch(batch):
                if action == "label":
                    item = invert_label(item, image_key=self.image_key, label_key=self.label_key)
                orig = item[bk].meta.get("filename_or_obj", "unknown.nii.gz")
                if isinstance(orig, str):
                    origstem = orig.split("/")[-1].split(".")[0]
                else:
                    origstem = "unknown"
                item[bk].meta["filename_or_obj"] = f"{epoch}_{origstem}_{k}.nii.gz"
                if item[bk].dtype == torch.bfloat16:
                    item[bk] = item[bk].to(torch.float32)
            SaveImage(output_dir=self.save_path, output_postfix="", separate_folder=False)(item[bk])
        
        _process(action)

    def process_action(self, trainer, batch, outputs):
        for k, action in outputs.items():
            if not isinstance(action, tuple):
                continue
            self._process_action(k, batch, trainer.current_epoch, action)


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.on_train_end:
            return
        if batch_idx + 1 != trainer.num_training_batches:
            return
        self.process_action(trainer, batch, outputs)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        if not self.on_val_end:
            return
        if batch_idx + 1 != trainer.num_training_batches:
            return
        self.process_action(trainer, batch, outputs)

    def on_test_batch_end(
        self, trainer, pl_module, outputs: dict[str, tuple[str, MetaTensor]], batch, batch_idx, dataloader_idx=0
    ):
        if not self.on_test_end:
            return
        # これは推論/保存も兼ねているのですべてのパッチに対して行う.
        self.process_action(trainer, batch, outputs)