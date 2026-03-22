from abc import ABC, abstractmethod
import warnings

import experiments.config
from experiments.config import channel_dim, size_dim
from experiments.utils import identity
from torch import Tensor
from typing import Callable

class Assertion(ABC):
    def __init__(self, *args, msg: str | None = None, **kwargs): 
        self.msg = msg

    def __call__(self, *args, **kwargs): 
        if experiments.config.assertion:
            self.call(*args, **kwargs)

    @abstractmethod
    def call(self, *args, **kwargs):
        ...

class DumbAssertion(Assertion):
    def call(self, *args, **kwargs):
        return


class Assert(Assertion):
    def __init__(self, *, msg: str | None = None):
        super().__init__(msg=msg)

    def call(self, a):
        assert a, self.msg

class AssertEq(Assert):
    def call(self, a, b):
        super().call(a == b)

class AssertNe(Assert):
    def call(self, a, b):
        return super().call(a != b)

class SeqAssertion(Assertion):
    def __init__(self, *assertions):
        super().__init__()
        self.assertions = assertions

    def call(self, *args, **kwargs):
        for assertion in self.assertions:
            assertion(*args, **kwargs)


class AssertShape(Assertion):
    def __init__(
        self,
        input_shape: int | tuple[int, ...] | Callable[[int], None] | None = None,
        output_shape: int | tuple[int, ...] | Callable[[int], None] | None = None,
        shape_fn: (
            Callable[[int | tuple[int, ...]], int | tuple[int, ...]] | None
        ) = None,
        dim: int | slice = slice(None),
        *,
        msg: str | None = None
    ):
        super().__init__(msg=msg)
        if (
            isinstance(input_shape, (int, tuple))
            or isinstance(output_shape, (int, tuple))
        ) and shape_fn is not None:
            warnings.warn(
                """Either input_shape or output_shape is provided alongside shape_fn.
shape_fn is intended to be used when you know neither input_shape nor output_shape.
Try to set both input_shape and output_shape."""
            )
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.shape_fn = shape_fn
        self.dim = dim

    def _assert(self, x: Tensor, y: Tensor):
        if self.input_shape is not None:
            if callable(self.input_shape):
                self.input_shape(x.shape[self.dim])
            else:
                AssertEq()(self.input_shape, x.shape[self.dim])
        if self.output_shape is not None:
            if callable(self.output_shape):
                self.output_shape(y.shape[self.dim])
            else:
                AssertEq()(self.output_shape, y.shape[self.dim])
        if self.shape_fn is not None:
            AssertEq()(self.shape_fn(x.shape[self.dim]), y.shape[self.dim])

    def call(self, *args: Tensor, **_):
        assert len(args) >= 2, f"Assert Shape got {len(args)} arguments"
        x = args[0]
        y = args[-1]
        self._assert(x, y)


class AssertChannel(AssertShape):
    def __init__(self, input_shape=None, output_shape=None, shape_fn=None, *, msg=None):
        super().__init__(input_shape, output_shape, shape_fn, dim=channel_dim, msg=msg)


class AssertSize(AssertShape):
    def __init__(self, input_shape=None, output_shape=None, shape_fn=None, *, msg=None):
        super().__init__(input_shape, output_shape, shape_fn, dim=size_dim, msg=msg)


class AssertNoShapeChange(AssertShape):
    def __init__(self, dim=slice(None), *, msg = None):
        super().__init__(None, None, identity, dim=dim, msg=msg)


class AssertNoChannelChange(AssertNoShapeChange):
    def __init__(self, *, msg=None):
        super().__init__(dim=channel_dim, msg=msg)


class AssertNoSizeChange(AssertNoShapeChange):
    def __init__(self, *, msg=None):
        super().__init__(dim=size_dim, msg=msg)
