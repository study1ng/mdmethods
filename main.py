import importlib
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lib", type=str)
    parser.add_argument("method", type=str, choices=["analyze", "prune", "preprocess", "train", "inference"])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()
    experiment = importlib.import_module(f"experiments.{parsed.lib}")
    assert hasattr(experiment, parsed.method), f"experiments.{parsed.lib} doesn't have method {parsed.method}"
    getattr(experiment, parsed.method)(parsed.arguments)
    