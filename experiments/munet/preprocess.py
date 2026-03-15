from .. import ImageLabelPreprocessor
import argparse
from monai.transforms import Compose, SaveImaged
from ..preprocess import load_transformd, planned_transformd
from ..utils import resolved_path, loaded_json
from ..config import image_key, label_key

class PlannedPreprocessor(ImageLabelPreprocessor):
    def get_argument_parser(self) -> argparse.ArgumentParser:
        parser = super().get_argument_parser()
        parser.add_argument("plan_path", type=resolved_path)

    def parse_args(self, args):
        super().parse_args(args)
        self.image_save_path = self.args.save_path / self.args.dataset / image_key
        self.label_save_path = self.args.save_path / self.args.dataset / label_key
        self.image_save_path.mkdir(exist_ok = True)
        self.label_save_path.mkdir(exist_ok = True)

    def preprocess_transform(self):
        assert self.args.plan_path.exists(), f"plan file {self.args.plan_path} not exists"
        plan = loaded_json(self.args.plan_path)
        return Compose(
            [
                load_transformd(self.keys),
                planned_transformd(plan, self.image_key, self.label_key),
                SaveImaged(self.image_key, output_dir=self.args.image_save_path, separate_folder=False, output_postfix=""),
                SaveImaged(self.label_key, output_dir=self.args.label_save_path, separate_folder=False, output_postfix="")
            ]
        )
