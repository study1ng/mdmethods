from pathlib import Path
import json

from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    RandZoomd,
    RandRotated,
    RandFlipd,
)
from torch import autocast, nn
import torch
from methods.structures.nets.UBiMambaEnc_3d import UNetEnc, InitWeights_He
import numpy as np
import torch.amp
from utils import nowstring
import logging
import time
import sys
from preprocess import load_transformd, padded_crop_wrapper
from methods.structures.mim import MIMUNet, generate_mask


def augmentation_transforms(keys, plan):
    # 回転処理のために余分にパディングしておく
    patch_size = plan["patch_size"]
    do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > 3
    if do_dummy_2d_data_aug:
        rotation_for_DA = {
            "range_x": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
            "range_y": (0, 0),
            "range_z": (0, 0),
        }
        min_zoom = [0.8, 1.0, 1.0]
        max_zoom = [1.2, 1.0, 1.0]
    else:
        rotation_for_DA = {
            "range_x": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
            "range_y": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
            "range_z": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
        }
        min_zoom = 0.8
        max_zoom = 1.2
    composelist = [
        load_transformd(keys),
        padded_crop_wrapper(
            keys,
            patch_size,
            [
                RandRotated(keys, **rotation_for_DA, prob=0.2),
                RandZoomd(keys, prob=0.2, min_zoom=min_zoom, max_zoom=max_zoom),
                RandFlipd(keys, prob=0.5, spatial_axis=0),
                RandFlipd(keys, prob=0.5, spatial_axis=1),
                RandFlipd(keys, prob=0.5, spatial_axis=2),
            ],
        ),
    ]
    return Compose(composelist)


def ssl(
    preprocessed: Path,
    pretrained: Path,
    dataset: str,
    plan_path: str,
    device: str | None = None,
    checkpoint: str | None = None,
):
    pimgs = preprocessed / dataset
    plan = json.load(Path(plan_path).open())
    ptdir = pretrained / dataset / nowstring()
    ptdir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
        handlers=[
            logging.FileHandler(ptdir / "pretraining.log"),  # ファイルへの出力
            logging.StreamHandler(sys.stdout),  # 標準出力への出力
        ],
    )

    device = torch.device(device)
    assert pimgs.exists(), f"the preprocessed img dir {pimgs} do not exists"
    imgs = [
        {"image": img, "name": img.name.split(".")[0].split("_")[0]}
        for img in pimgs.iterdir()
        if img.suffix == ".gz"
    ]
    keys = ["image"]
    data = Dataset(imgs, augmentation_transforms(keys, plan))
    loader = DataLoader(
        data, batch_size=plan["configurations"]["3d_fullres"]["batch_size"], shuffle=True, num_workers=4
    )

    model = UNetEnc(plan, 1, 1).to(device)
    net = MIMUNet(model).to(device)
    net.apply(InitWeights_He(1e-2))
    logging.info(net)
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=4e-4,
        eps=1e-5,
        weight_decay=1e-1,
        betas=(0.9, 0.95),
    )
    total_steps = 300
    scheduler_steps = total_steps * len(loader)
    warmup_steps = int(0.1 * scheduler_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-10,
                end_factor=1.0,
                total_iters=warmup_steps,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=scheduler_steps - warmup_steps, eta_min=1e-5
            ),
        ],
        milestones=[warmup_steps],
    )
    loss = nn.L1Loss(reduction="none")
    gradscaler = torch.amp.GradScaler("cuda")

    strides = np.array(plan["pool_op_kernel_sizes"])
    mask_shape = np.prod(strides, axis=0)
    mask_ratio = 0.6
    if checkpoint is not None:
        loaded = torch.load(Path(checkpoint), weights_only=True)
        net.load_state_dict(loaded["model"])
        optimizer.load_state_dict(loaded["optim"])
        gradscaler.load_state_dict(loaded["grad"])
        scheduler.load_state_dict(loaded["scheduler"])
        current_epoch = loaded["epoch"] + 1
    else:
        current_epoch = 1
    net.train()
    for epoch in range(current_epoch, total_steps + 1):
        logging.info(f"epoch: {epoch}")
        start_time = time.perf_counter()
        # train step
        losses = []
        for batch in loader:
            image = batch["image"]
            image = image.to(device)
            mask = generate_mask(image, mask_shape, mask_ratio, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device.type, enabled=True):
                mask = mask.type_as(image)
                output = net(image, mask)
                l = loss(output, image)
                l = l * mask
                l = l.sum() / (mask.sum() * image.shape[1] + 1e-5)
            gradscaler.scale(l).backward()
            gradscaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            gradscaler.step(optimizer)
            gradscaler.update()
            l = l.detach().cpu().numpy()
            logging.info(f"loss: {l}")
            losses.append(l)
            scheduler.step()
            logging.info(f"lr: {scheduler.get_last_lr()}")
        logging.info(f"mean_loss_for_epoch: {np.mean(losses, axis=0)}")
        logging.info(f"epoch {epoch} end")
        logging.info(f"time: {time.perf_counter() - start_time}s")
        if epoch % 10 == 0:
            # モデルを保存する.
            dic = {
                "model": net.state_dict(),
                "optim": optimizer.state_dict(),
                "grad": gradscaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            }
            torch.save(dic, ptdir / f"model_{epoch}.pth")

    # モデルを保存する.
    dic = {
        "model": net.state_dict(),
        "optim": optimizer.state_dict(),
        "grad": gradscaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }
    torch.save(dic, ptdir / "model_latest.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("plan")
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("-c", "--checkpoint", default=None)
    p = parser.parse_args()

    preprocessed = Path("data/preprocessed").expanduser().resolve()
    pretrained = Path("data/pretrained").expanduser().resolve()
    ssl(preprocessed, pretrained, p.dataset, p.plan, p.device, p.checkpoint)
