import argparse
from abc import ABC, abstractmethod

class ArgumentAdaptor(ABC):
    def __init__(self, args: str, meta: argparse.Namespace):
        """A class get command line arguments and process something

        Parameters
        ----------
        args : str
            the command line arguments
        meta : Namespace
            some meta information come from main.py
        """
        self.meta = meta
        self.parse_args(args)

    def get_argument_parser(self) -> argparse.ArgumentParser:
        """get argument parser

        Returns
        -------
        argparse.ArgumentParser

        Examples
        --------
        This class is intended to be inherited to add some arguments.
        ```
        def get_argument_parser(self):
            parser = super().get_argument_parser()
            parser.add_argument("some_path")
            return parser
        """
        parser = argparse.ArgumentParser()
        return parser

    def parse_args(self, args: list[str]):
        """parse arguments and process something. Some properties you want to calculate before __call__ but need the args, you can calculate it here

        Parameters
        ----------
        args : list[str]
            arguments
        """
        parser = self.get_argument_parser()
        self.args = parser.parse_args(args)

    @abstractmethod
    def __call__(self):
        ...