from experiments import ImageLabelPreprocessor
import argparse
from monai.transforms import Compose, SaveImaged
from experiments.preprocess import load_transformd, planned_transformd
from experiments.utils import resolved_path
from experiments.config import image_key, label_key
from experiments.utils import ensure_dir_new, assert_file_exist
from experiments.plan import Plan


class PlannedPreprocessor(ImageLabelPreprocessor):
    def get_argument_parser(self) -> argparse.ArgumentParser:
        parser = super().get_argument_parser()
        parser.add_argument("-p", "--plan_path", type=resolved_path)
        return parser

    def parse_args(self, args):
        super().parse_args(args)
        self.image_save_path = self.args.save_path / image_key
        self.label_save_path = self.args.save_path / label_key
        ensure_dir_new(self.image_save_path)
        ensure_dir_new(self.label_save_path)
        assert_file_exist(self.args.plan_path)
        self.plan = Plan(self.args.plan_path)

    def preprocess_transform(self):
        return Compose(
            [
                load_transformd([image_key, label_key]),
                planned_transformd(self.plan, [image_key], [label_key]),
                SaveImaged([image_key], output_dir=self.image_save_path, separate_folder=False, output_postfix=""),
                SaveImaged([label_key], output_dir=self.label_save_path, separate_folder=False, output_postfix="")
            ]
        )
