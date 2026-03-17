from experiments import ImageOnlyPreprocessor
import argparse
from monai.transforms import Compose, SaveImaged
from experiments.preprocess import load_transformd, planned_transformd
from experiments.utils import resolved_path, loaded_json
from experiments.config import image_key

class PlannedSSLPreprocessor(ImageOnlyPreprocessor):
    def __init__(self, args):
        super().__init__(args)
        self.image_key = [image_key]
        self.save_path = self.args.save_path

    def get_argument_parser(self) -> argparse.ArgumentParser:
        parser = super().get_argument_parser()
        parser.add_argument("plan_path", type=resolved_path)

    def preprocess_transform(self):
        assert self.args.plan_path.exists(), f"plan file {self.args.plan_path} not exists"
        plan = loaded_json(self.args.plan_path)
        return Compose(
            [
                load_transformd(self.image_key),
                planned_transformd(plan, self.image_key),
                SaveImaged(self.image_key, output_dir=self.save_path, separate_folder=False, output_postfix="")
            ]
        )
