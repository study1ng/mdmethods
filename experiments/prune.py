from pathlib import Path
import json
from shutil import rmtree
from abc import abstractmethod, ABC
from . import AnalyzedData
import argparse
from .utils import resolved_path, loaded_json

class Pruner(ABC):
    def __init__(self, args: list[str]):
        self.parse_args(args)

    def parse_args(self, args: list[str]):
        parser = self.get_argument_parser()
        self.args = parser.parse_args(args)
        self.analyzed = loaded_json(self.args.analyzed_path)

    def get_argument_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument("analyzed_path", type=resolved_path)
        parser.add_argument("save_path", type=resolved_path)
        return parser

    @abstractmethod
    def need_prune(self, *args, **kwargs):
        ...

    def link_or_prune(self, data: AnalyzedData, save_path: Path) -> bool:
        if self.need_prune(data):
            return False
        save_path.unlink(missing_ok = True)
        save_path.symlink_to(resolved_path(data.path))
        return True

    def __call__(self):
        if self.args.save_path.exists():
            rmtree(self.args.save_path)
        self.args.save_path.mkdir(parents=True, exist_ok=True)
        analyzed = [AnalyzedData(i) for i in self.analyzed]
        idx = 0
        print("processing", len(analyzed), "volumes")
        while len(analyzed) != 0:
            data = analyzed.pop()
            idx += self.link_or_prune(data = data, save_path = self.args.save_path / f"{idx}.nii.gz")
        print(idx, "volumes will be used")

class SpacingShapeStrictPruner(Pruner):
    def __init__(self, args: list[str], default_allowed_spacing_factor: float = 1.5, default_allowed_shape_factor: float = 1.5):
        self.default_allowed_spacing_factor=default_allowed_spacing_factor
        self.default_allowed_shape_factor = default_allowed_shape_factor
        super().__init__(args)

    def get_argument_parser(self):
        parser = super().get_argument_parser()
        parser.add_argument("plan", type=Path)
        parser.add_argument("--allowed_spacing_factor", type=float, default=self.default_allowed_spacing_factor)
        parser.add_argument("--allowed_shape_factor", type=float, default=self.default_allowed_shape_factor)
        return parser
    
    def parse_args(self, args):
        super().parse_args(args)
        self.plan = loaded_json(self.args.plan)

    def need_prune(
        self,
        data: AnalyzedData
    ) -> bool:
        allowed_spacing_factor = self.args.allowed_spacing_factor
        allowed_shape_factor = self.args.allowed_shape_factor
        splan = self.plan["configurations"]["3d_fullres"]
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

