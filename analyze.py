from pathlib import Path
import monai.data
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    CropForeground,
    EnsureType,
    Compose,
)
from tqdm import tqdm
import concurrent.futures
import json
import torch
import numpy as np

def _analyze(pth: Path):
    # 画素値の最大値, 最小値, 平均値, 中間値, 0.5パーセンタイル値, 99.5パーセンタイル値, 5パーセンタイル値, 95パーセンタイル値
    # 標準偏差, スペーシング, クロップ後の形状を出力する.
    transforms = Compose(
        [
            LoadImage(),
            EnsureChannelFirst(),
            EnsureType(dtype=torch.float32, track_meta=True),
            CropForeground(
                select_fn = lambda x: x >= -900,
                margin=10,
                allow_smaller=False,
            ),
        ]
    )
    transformed: monai.data.MetaTensor = transforms(pth)
    percentiles = np.quantile(transformed.numpy(), [0., 0.005, 0.050, 0.50, 0.950, 0.995, 1.])
    return {
        "path": str(pth),
        "max": percentiles[-1].item(),
        "min": percentiles[0].item(),
        "mean": transformed.mean().item(),
        "median": percentiles[3].item(),
        "p005": percentiles[1].item(),
        "p995": percentiles[5].item(),
        "p050": percentiles[2].item(),
        "p950": percentiles[4].item(),
        "std": transformed.std().item(),
        "spacing_original": [i.item() for i in transformed.pixdim],
        "cropped_shape": tuple(transformed.shape),
    }


def analyze(pth: Path):
    max_workers = 12
    analyzed = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_analyze, p) for p in pth.iterdir() if p.suffix == ".gz"]
        try:
            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                analyzed.append(future.result())
        except Exception as e:
            executor.shutdown(wait=False, cancel_futures=True)
            raise e
    return analyzed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("analyzed")
    args = parser.parse_args()
    tar = Path(args.target).expanduser().resolve()
    assert tar.exists(), f"{tar} was passed as target but not exists"
    analyzed = analyze(tar)
    
    with Path(args.analyzed).expanduser().resolve().open("w") as f:
        json.dump(analyzed, f)
