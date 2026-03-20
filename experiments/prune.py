from pathlib import Path
from abc import abstractmethod
from experiments import AnalyzedData
import argparse
from experiments.argument_adaptor import ArgumentAdaptor
from experiments.utils import (
    resolved_path,
    loaded_json,
    filekey,
    assert_file_exist,
    ensure_dir_new,
    set_symln,
)
from typing import Sequence
import concurrent.futures
from tqdm import tqdm
from experiments.plan import Plan



class Pruner(ArgumentAdaptor):
    """Prune path in analyzed_path and save them use symbolick link
    """
    @classmethod
    def is_target(cls, p: Path) -> bool:
        """whether p need to be processed by this Pruner

        Parameters
        ----------
        p : Path
            target path

        Returns
        -------
        bool
            if p is target
        """
        return p.suffix == ".gz"

    def get_argument_parser(self) -> argparse.ArgumentParser:
        parser = super().get_argument_parser()
        parser.add_argument("-w", "--workers", type=int, default=4)
        parser.add_argument("analyzed_path", type=resolved_path)
        parser.add_argument(
            "target_paths_and_save_paths", type=resolved_path, nargs="+"
        )
        return parser

    def parse_args(self, args: list[str]):
        parser = self.get_argument_parser()
        self.args = parser.parse_args(args)
        self.analyzed = loaded_json(self.args.analyzed_path)
        self.target_paths: list[Path] = self.args.target_paths_and_save_paths[::2]
        self.save_paths: list[Path] = self.args.target_paths_and_save_paths[1::2]
        for save_path in self.save_paths:
            ensure_dir_new(save_path)
        assert len(self.target_paths) == len(
            self.save_paths
        ), "target_paths.len != save_paths.len"

    @abstractmethod
    def need_prune(self, data: AnalyzedData) -> bool: 
        """if data is needed to be pruned

        Parameters
        ----------
        data : AnalyzedData
            

        Returns
        -------
        bool
        """
        ...

    @staticmethod
    def _all_has_same_key(*dicts) -> bool:
        keys = [set(dic.keys()) for dic in dicts]
        for i in keys[1:]:
            if keys[0] != i:
                return False
        return True

    def _construct_processing_dict(
        self,
    ) -> dict[str, tuple[AnalyzedData, Sequence[tuple[Path, Path]]]]:
        """
        
        Returns
        -------
        dict[str, tuple[AnalyzedData, Sequence[tuple[Path, Path]]]]
            {"name": (analyzed_data, (target_path, save_path)*), ...} という形式の辞書を返す.
        """
        analyzed = {}
        for i in self.analyzed:
            an = AnalyzedData(i)
            analyzed[an.filekey] = an
        rets = [analyzed]
        for target_path, save_path in zip(self.target_paths, self.save_paths):
            paired = {}
            for i in target_path.iterdir():
                if not self.is_target(i):
                    continue
                fk = filekey(i)
                if paired.get(fk):
                    raise Exception(f"{i} and {paired[fk][0]} has same filekey")
                paired[fk] = (i, save_path / f"{fk}.nii.gz")
            rets.append(paired)
        assert self._all_has_same_key(*rets), f"{[ret.keys() for ret in rets]}"
        ret = {key: tuple(d[key] for d in rets) for key in rets[0].keys()}
        return ret

    def __call__(self):
        processing_dict = self._construct_processing_dict()
        print("processing", len(processing_dict), "entries")
        count = 0
        tasks = list(processing_dict.values())
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.workers) as executor:
            futures = [executor.submit(self._process_entry, *task) for task in tasks]
            try:
                for future in tqdm(
                    concurrent.futures.as_completed(futures), total=len(futures)
                ):
                    count += future.result()
            except Exception as e:
                executor.shutdown(wait=False, cancel_futures=True)
                raise e
        print(count, "entries will be used")

    def _process_entry(
        self, analyze: AnalyzedData, *target_and_save: tuple[Path, Path]
    ) -> bool:
        """process one task

        Parameters
        ----------
        analyze : AnalyzedData


        Returns
        -------
        bool
            whether it was not pruned
        """
        if self.need_prune(analyze):
            return False
        for t in target_and_save:
            target, save = t
            assert_file_exist(target)
            set_symln(from_=target, to=save)
        return True


class SpacingShapeStrictPruner(Pruner):
    """Pruner which use plan to check spacing and shape

    Parameters
    ----------
    default_allowed_spacing_factor: float = 1.5
        the default spacing strictness factor, the higher is strict
    default_allowed_shape_factor: float = 1.5,
        the default shape strictness factor, the higher is strict
    """
    def __init__(
        self,
        args: list[str],
        meta,
        default_allowed_spacing_factor: float = 1.5,
        default_allowed_shape_factor: float = 1.5,
    ):
        self.default_allowed_spacing_factor = default_allowed_spacing_factor
        self.default_allowed_shape_factor = default_allowed_shape_factor
        super().__init__(args, meta)

    def get_argument_parser(self):
        parser = super().get_argument_parser()
        parser.add_argument("-p", "--plan_path", type=Path)
        parser.add_argument(
            "--allowed_spacing_factor",
            type=float,
            default=self.default_allowed_spacing_factor,
        )
        parser.add_argument(
            "--allowed_shape_factor",
            type=float,
            default=self.default_allowed_shape_factor,
        )
        return parser

    def parse_args(self, args):
        super().parse_args(args)
        self.plan = Plan(self.args.plan_path)
        self.plan_spacing = self.plan.spacing
        self.plan_shape = self.plan.patch_size
        self.spacing_criteria = [
            i * self.args.allowed_spacing_factor for i in self.plan_spacing
        ]
        self.shape_criteria = [int(i * self.args.allowed_shape_factor) for i in self.plan_shape]


    def need_prune(self, data: AnalyzedData) -> bool:
        prune = (
            data.spacing_x >= self.spacing_criteria[0]
            or data.spacing_y >= self.spacing_criteria[1]
            or data.spacing_z >= self.spacing_criteria[2]
            or data.shape_x * data.spacing_x <= self.shape_criteria[0] * self.plan_spacing[0]
            or data.shape_y * data.spacing_y <= self.shape_criteria[1] * self.plan_spacing[1]
            or data.shape_z * data.spacing_z <= self.shape_criteria[2] * self.plan_spacing[2]
        )  # 各軸のボクセル間隔のいずれかがcriteriaより大きい場合か, 形状のいずれかがcriteriaより小さい場合に刈り取られる
        return prune


class NoPruner(Pruner):
    """Pruner which don't prune
    """
    def need_prune(self, data):
        # no prune
        return False
