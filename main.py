import importlib
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lib", type=str)
    parser.add_argument(
        "method",
        type=str,
        choices=["analyze", "prune", "preprocess", "train", "inference"],
    )
    parser.add_argument("--experiment_name", type=str, default=None)
    parsed, unknown = parser.parse_known_args()
    libname = (
        f"experiments.{parsed.lib}"
        if parsed.experiment_name is None
        else f"experiments.{parsed.lib}.experiments.{parsed.experiment_name}"
    )
    experimentlib = importlib.import_module(libname)
    assert hasattr(
        experimentlib, parsed.method
    ), f"{libname} doesn't have method {parsed.method}"
    getattr(experimentlib, parsed.method)(unknown)
