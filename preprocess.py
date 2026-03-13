from monai.transforms import (
    Compose,
    SaveImaged,
)
from monai.data import Dataset, DataLoader
from pathlib import Path
import argparse
import json
import tqdm
from preprocess import load_transformd, planned_transformd

def transform(plan_path: str, save_path: Path):
    plan_path = Path(plan_path).expanduser().resolve()
    assert plan_path.exists(), f"plan file {plan_path} not exists"
    plan = json.load(plan_path.open())
    keys = ["image"]
    return Compose(
        [
            load_transformd(keys),
            planned_transformd(keys, plan),
            SaveImaged(keys, output_dir=save_path, separate_folder=False, output_postfix="")
        ]
    )


def preprocess(
    raw: Path, preprocessed: Path, dataset: str, plan_path: str, worker: int
):
    if worker is None:
        worker = 1
    rdir: Path = raw / dataset
    assert rdir.exists(), f"the raw directory {rdir} not exists"
    pdir: Path = preprocessed / dataset
    pdir.mkdir(exist_ok=True)

    rimgs = rdir
    pimgs = pdir

    assert rimgs.exists(), f"the raw images dir {rimgs} not exists"
    pimgs.mkdir(exist_ok=True)

    rimgd = {}
    for rimg in rimgs.iterdir():
        assert not rimg.is_dir(), f"a dir {rimg} in raw image dir {rimgs}"
        # 仕様として, stem部分の_以降はコメントとして無視して, 初めの_以前のみをファイル名として扱う
        name = rimg.name.split(".")[0].split("_")[0]
        rimgd[name] = rimg

    nrimg = set(rimgd.keys())
    rname = nrimg
    rpaird = [{"image": rimgd[name], "name": name} for name in rname]
    transforms = transform(plan_path, save_path=pdir)
    dataset = Dataset(rpaird, transforms)
    for _ in tqdm.tqdm(
        DataLoader(
            dataset, num_workers=worker, batch_size=1, collate_fn=lambda x: x[0]
        ),
        dynamic_ncols=True,
    ):
        pass

if __name__ == "__main__":
    raw = Path("./data/raw").expanduser().resolve()
    preprocessed = (
        Path("./data/preprocessed").expanduser().resolve()
    )

    assert raw.exists(), f"the raw root {raw} not exists"
    if not preprocessed.exists():
        preprocessed.mkdir(parents=True)

    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("plan")
    p.add_argument("-w", "--worker", type=int, default=1)
    parsed = p.parse_args()
    preprocess(raw, preprocessed, parsed.dataset, parsed.plan, parsed.worker)
