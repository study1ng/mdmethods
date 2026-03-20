import argparse
from abc import ABC, abstractmethod

class ArgumentAdaptor(ABC):
    def __init__(self, args: list[str], meta):
        self.meta = meta
        self.parse_args(args)

    def get_argument_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        return parser

    def parse_args(self, args: list[str]):
        parser = self.get_argument_parser()
        self.args = parser.parse_args(args)

    @abstractmethod
    def __call__(self):
        ...