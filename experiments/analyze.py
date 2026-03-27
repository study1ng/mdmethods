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
from abc import abstractmethod
from experiments.utils.fsutils import resolved_path
import argparse
from experiments import ArgumentAdaptor


class Analyzer(ArgumentAdaptor):
    """Creating the summary of data"""

    def get_argument_parser(self) -> argparse.ArgumentParser:
        parser = super().get_argument_parser()
        parser.add_argument("-w", "--workers", default=4, type=int)
        return parser

    @abstractmethod
    def analyze(self, p: Path) -> dict: 
        """analyze one data and return its summary

        Parameters
        ----------
        p : Path

        Returns
        -------
        dict

        """
        ...

    @abstractmethod
    def get_target_paths(self) -> list[Path]: 
        """Get the path list which need to be analyzed

        Returns
        -------
        list[Path]
        """
        ...

    def __call__(self):
        max_workers = self.args.workers
        analyzed = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [
                executor.submit(self.analyze, p) for p in self.get_target_paths()
            ]
            try:
                for future in tqdm(
                    concurrent.futures.as_completed(futures), total=len(futures)
                ):
                    analyzed.append(future.result())
            except Exception as e:
                executor.shutdown(wait=False, cancel_futures=True)
                raise e
        self.after_analyze(analyzed)

    @abstractmethod
    def after_analyze(self, analyzed: list[dict]): 
        """after all data was summarized, this function would be called. you can save data to files

        Parameters
        ----------
        analyzed : list[dict]
            the summarized data
        """
        ...


class CTAnalyzer(Analyzer):
    """Analyze Nifti data and return max, min, mean, median, 0.5 percentile, 99.5 percentile, 5 percentile, 95 percentile, std, spacing, shape after crop

    Parameters
    ----------
    Analyzer : _type_
        _description_
    """
    def parse_args(self, args):
        super().parse_args(args)
        self.target = self.args.target

    def get_target_paths(self):
        return [p for p in self.target.iterdir() if p.suffix == ".gz"]

    def analyze(self, p: Path) -> dict:
        # 画素値の最大値, 最小値, 平均値, 中間値, 0.5パーセンタイル値, 99.5パーセンタイル値, 5パーセンタイル値, 95パーセンタイル値
        # 標準偏差, スペーシング, クロップ後の形状を出力する.
        transforms = Compose(
            [
                LoadImage(),
                EnsureChannelFirst(),
                EnsureType(dtype=torch.float32, track_meta=True),
                CropForeground(
                    select_fn=lambda x: x >= -900,
                    margin=10,
                    allow_smaller=False,
                ),
            ]
        )
        transformed: monai.data.MetaTensor = transforms(p)
        percentiles = np.quantile(
            transformed.numpy(), [0.0, 0.005, 0.050, 0.50, 0.950, 0.995, 1.0]
        )
        return {
            "path": str(p),
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

    def get_argument_parser(self):
        parser = super().get_argument_parser()
        parser.add_argument("target", type=resolved_path)
        parser.add_argument("analyzed", type=resolved_path)
        return parser

    def after_analyze(self, analyzed: list[dict]):
        """dump to json

        Parameters
        ----------
        analyzed : list[dict]
        """
        with self.args.analyzed.open("w") as f:
            json.dump(analyzed, f)
