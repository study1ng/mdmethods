import importlib
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str, choices=["preprocess", "prune", "train", "inference"])
    parser.add_argument("lib", type=str)
    parser.add_argument("arguments", nargs="argparse.REMAINDER")
    parsed = parser.parse_args()
    experiment = importlib.import_module(f"experiments.{parsed.lib}")
    match parsed.method:
        case "preprocess":
            func = experiment.preprocess
        case "prune":
            func = experiment.prune
        case "train":
            func = experiment.train
        case "inference":
            func = experiment.inference
    func(parsed.arguments)