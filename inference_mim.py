from mim import MIMUNet, generate_mask, UNetEnc
from pathlib import Path
import json
import torch
import numpy as np
from monai.transforms import (
    Compose,
    RandSpatialCropd,
    SaveImage,
)
from monai.data import MetaTensor
from preprocess import load_transformd, planned_transformd

def inference(
    checkpoint: Path,
    plan_path: str,
    image_path: Path,
    save_path: Path,
    patch_count: int,
    device: str | None = None,
):
    # image_count枚くらいパッチに切り出して, ランダムなマスクを生成して, save_pathに保存する

    plan = json.load(Path(plan_path).expanduser().resolve().open())
    jplan = plan["configurations"]["3d_fullres"]
    model = UNetEnc.from_plan(plan, 1, 1).to(device)
    net = MIMUNet(model).to(device)
    loaded = torch.load(
        checkpoint,
        weights_only=True,
    )
    net.load_state_dict(loaded["model"])

    strides = np.array(jplan["pool_op_kernel_sizes"])
    mask_shape = np.prod(strides, axis=0)
    mask_ratio = 0.6    
    keys = ["image"]
    load_transform = Compose(
        [
            load_transformd(keys),
            planned_transformd(keys, plan),
        ]
    )
    loaded = load_transform({"image": image_path})
    cropper = RandSpatialCropd(keys, jplan["patch_size"], random_center=True, random_size=False)
    net.eval()        
    splan = plan["foreground_intensity_properties_per_channel"]["0"]
    std: float = splan["std"]  # standard deviation of whole training dataset
    mean: float = splan["mean"]  # average of whole training dataset
    for i in range(patch_count):
        cropped = cropper(loaded)["image"].unsqueeze(0).to(device)
        mask = generate_mask(cropped, mask_shape, mask_ratio, device)
        masked = cropped * (1 - mask)
        with torch.no_grad():
            restored = net(cropped, mask)


        # zスコア正規化を行う前の状態にする
        zcropped = cropped * std + mean
        zmasked = masked * std + mean
        zrestored = restored * std + mean
        zmasked[mask.bool()] = -1000

        mcropped = MetaTensor(zcropped.squeeze(0).cpu(), cropped.affine, cropped.meta)
        mmasked = MetaTensor(zmasked.squeeze(0).cpu(), cropped.affine, cropped.meta)
        mrestored = MetaTensor(zrestored.squeeze(0).cpu(), cropped.affine, cropped.meta)

        SaveImage(output_dir=save_path, output_postfix=f"{i}_cropped", separate_folder=False)(mcropped)
        SaveImage(output_dir=save_path, output_postfix=f"{i}_masked", separate_folder=False)(mmasked)
        SaveImage(output_dir=save_path, output_postfix=f"{i}_restored", separate_folder=False)(mrestored)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=lambda x: Path(x).expanduser().resolve())
    parser.add_argument("plan_path")
    parser.add_argument("image_path", type=lambda x: Path(x).expanduser().resolve())
    parser.add_argument("save_path", type=lambda x: Path(x).expanduser().resolve())
    parser.add_argument("patch_count", type=int)
    parser.add_argument("-d", "--device", default="cuda:0")
    args = parser.parse_args()

    inference(args.checkpoint, args.plan_path, args.image_path, args.save_path, args.patch_count, args.device)
