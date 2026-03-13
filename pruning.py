from pathlib import Path
import json
from datatypes import AnalyzedData
from shutil import rmtree


def need_prune(
    data: AnalyzedData,
    plan,
    allowed_spacing_factor: float = 1.5,
    allowed_shape_factor: float = 1.5, # 許容される形状の最小値. 各軸パッチサイズの2倍以上ないと刈り取られるよう設定する.
) -> bool:
    splan = plan["configurations"]["3d_fullres"]
    plan_spacing = splan["spacing"][::-1]
    plan_shape = splan["patch_size"][::-1]
    spacing_criteria = [int(i * allowed_spacing_factor) for i in plan_spacing] # nnUNetのplanは[z, y, x]順なので[x, y, z]に直す.
    shape_criteria = [int(i * allowed_shape_factor) for i in plan_shape]
    prune = (
        data.spacing_x >= spacing_criteria[0]
        or data.spacing_y >= spacing_criteria[1]
        or data.spacing_z >= spacing_criteria[2]
        or data.shape_x * data.spacing_x <= shape_criteria[0] * plan_spacing[0]
        or data.shape_y * data.spacing_y <= shape_criteria[1] * plan_spacing[1]
        or data.shape_z * data.spacing_z <= shape_criteria[2] * plan_spacing[2]
    ) # 各軸のボクセル間隔のいずれかがcriteriaより大きい場合か, 形状のいずれかがcriteriaより小さい場合に刈り取られる
    return prune

def link_or_prune(data: AnalyzedData, plan, save_path: Path) -> bool:
    if need_prune(data, plan):
        return False
    save_path.unlink(missing_ok = True)
    save_path.symlink_to(Path(data.path).expanduser().resolve())
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("analyzed_path", type=lambda p: Path(p).expanduser().resolve())
    parser.add_argument("plan_path", type=lambda p: Path(p).expanduser().resolve())
    parser.add_argument("save_path", type=lambda p: Path(p).expanduser().resolve())
    args = parser.parse_args()
    rmtree(args.save_path)
    args.save_path.mkdir(parents=True, exist_ok=True)
    analyzed = json.load(args.analyzed_path.open())
    analyzed = [AnalyzedData(i) for i in analyzed]
    plan = json.load(args.plan_path.open())
    idx = 0
    print("processing", len(analyzed), "volumes")
    while len(analyzed) != 0:
        data = analyzed.pop()
        idx += link_or_prune(data, plan, args.save_path / f"{idx}.nii.gz")
    print(idx, "volumes will be used")