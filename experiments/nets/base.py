from abc import ABC, abstractmethod
from typing import Callable
import warnings
from torch import Tensor, nn
from experiments.plan import Plan
from experiments.utils import (
    assert_eq,
    assert_divisible,
    elementwise_mul,
    identity,
    repeat,
    scale_shape_fn,
    channel_dim,
    size_dim,
)
import experiments.config
import torch


class Assertion(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs): ...

    @abstractmethod
    def __call__(self, *args, **kwargs): ...


class DumbAssertion(Assertion):
    def __init__(self):
        pass

    def __call__(self, *_, **__):
        pass


class Assertions(Assertion):
    def __init__(self, *assertions):
        self.assertions = assertions

    def __call__(self, *args, **kwargs):
        if experiments.config.assertion:
            for assertion in self.assertions:
                assertion(*args, **kwargs)


class AssertShape(Assertion):
    def __init__(
        self,
        input_shape: int | tuple[int, ...] | None = None,
        output_shape: int | tuple[int, ...] | None = None,
        shape_fn: (
            Callable[[int | tuple[int, ...]], int | tuple[int, ...]] | None
        ) = None,
        dim: int | slice = slice(),
    ):
        if not (input_shape is None and output_shape is None) and shape_fn is not None:
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
        shape = x.shape[self.dim]
        if self.input_shape is not None:
            assert_eq(self.input_shape, shape)
        if self.output_shape is not None:
            assert_eq(self.output_shape, y.shape[self.dim])
        if self.shape_fn is not None:
            assert_eq(self.shape_fn(shape), y.shape[self.dim])

    def __call__(self, *args: Tensor, **_):
        assert len(args) >= 2, f"Assert Shape got {len(args)} arguments"
        x = args[0]
        y = args[-1]
        self._assert(x, y)


class AssertChannel(AssertShape):
    def __init__(self, input_shape=None, output_shape=None, shape_fn=None):
        super().__init__(input_shape, output_shape, shape_fn, dim=channel_dim)


class AssertSize(AssertShape):
    def __init__(self, input_shape=None, output_shape=None, shape_fn=None):
        super().__init__(input_shape, output_shape, shape_fn, dim=size_dim)


class AssertNoShapeChange(AssertShape):
    def __init__(self, dim=slice()):
        super().__init__(None, None, identity, dim=dim)


class AssertNoChannelChange(AssertNoShapeChange):
    def __init__(self):
        super().__init__(dim=channel_dim)


class AssertNoSizeChange(AssertNoShapeChange):
    def __init__(self):
        super().__init__(dim=size_dim)


class BaseUNetModule(nn.Module, ABC):
    """A marker class for this module's classes"""

    def __init__(self):
        super().__init__()
        self.assertions = DumbAssertion()

    def bound_assertion(self, assertions: Assertions):
        self.assertions = Assertions(self.assertions, assertions)

    def forward(self, *args, **kwargs):
        y = self._forward(*args, **kwargs)
        self.assertions(*args, y, **kwargs)
        return y

    @abstractmethod
    def _forward(self, *args, **kwargs): ...

    @abstractmethod
    def calculate_output_size(
        self, *input_size: tuple[int, ...] | tuple[tuple[int, ...], ...]
    ) -> tuple[int, ...] | tuple[tuple[int, ...], ...]: ...


class Block(BaseUNetModule):
    """Block, is a map which don't break spatial resolution

    Parameters
    ----------
    input_channel: int
        the input channel
    output_channel: int
        the output channel
    """

    def __init__(self, input_channel, output_channel):
        """Block, is a map which don't break spatial resolution

        Parameters
        ----------
        input_channel: int
            the input channel
        output_channel: int
            the output channel
        """
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.bound_assertion(
            Assertions(
                AssertNoSizeChange(),
                AssertChannel(self.input_channel, self.output_channel),
            )
        )

    def calculate_output_size(self, input_size):
        b, c, *s = input_size
        return (b, self.output_channel, *s)

    def forward(self, x):
        return super().forward(x)


class Pool(BaseUNetModule):
    def __init__(
        self,
        input_channel: int,
        output_channel,
        pool_stride,
    ):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.pool_stride = pool_stride
        self.bound_assertion(
            AssertChannel(self.input_channel, self.output_channel),
        )

    def forward(self, x):
        return super().forward(x)


class DownSampling(Pool):
    def __init__(self, input_channel, output_channel, pool_stride):
        super().__init__(input_channel, output_channel, pool_stride)
        self.bound_assertion(
            AssertSize(shape_fn=scale_shape_fn(scale=self.pool_stride, shrink=True))
        )

    def calculate_output_size(self, input_size):
        b, c, *s = input_size
        assert_eq(self.input_channel, c)
        s = assert_divisible(s, self.pool_stride)
        return (b, self.output_channel, *s)


class UpSampling(Pool):
    def __init__(self, input_channel, output_channel, pool_stride):
        super().__init__(input_channel, output_channel, pool_stride)
        self.bound_assertion(
            AssertSize(shape_fn=scale_shape_fn(scale=self.pool_stride))
        )


    def calculate_output_size(self, input_size):
        b, c, *s = input_size
        assert_eq(self.input_channel, c)
        s = elementwise_mul(s, self.pool_stride)
        return (b, self.output_channel, *s)


class Stage(BaseUNetModule):
    def __init__(
        self,
        input_channel: int,
        pool_channel: int,
        output_channel: int,
        pool_stride: int | tuple[int, ...],
    ):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.pool_channel = pool_channel
        self.pool_stride = pool_stride
        self.pool = self._build_pool()
        self.pool.bound_assertion(
            AssertChannel(self.input_channel, self.pool_channel),
        )
        self.block = self._build_block()
        self.block.bound_assertion(
            AssertChannel(output_shape=self.output_channel),
        )

    @abstractmethod
    def _build_pool(self) -> Pool: ...

    @abstractmethod
    def _build_block(self) -> Block: ...

    def _forward(self, x: Tensor) -> Tensor:
        m = self.pool(x)
        y = self.block(m)
        return y

    def calculate_output_size(self, input_size):
        m = self.pool.calculate_output_size(input_size)
        y = self.block.calculate_output_size(m)
        return y


class EncoderStage(Stage):
    def __init__(self, input_channel, pool_channel, output_channel, pool_stride):
        super().__init__(input_channel, pool_channel, output_channel, pool_stride)
        self.pool.bound_assertion(
            AssertSize(shape_fn=scale_shape_fn(self.pool_stride, shrink=True))
        )

    @abstractmethod
    def _build_pool(self) -> DownSampling: ...


class DecoderStage(Stage):
    def __init__(
        self, input_channel, pool_channel, skip_channel, output_channel, pool_stride
    ):
        self.skip_channel = skip_channel
        super().__init__(input_channel, pool_channel, output_channel, pool_stride)
        self.pool.bound_assertion(AssertSize(shape_fn=scale_shape_fn(self.pool_stride)))

    def _forward(self, x, skip):
        m = self.pool(x)
        AssertNoSizeChange()(m, skip)  # same spatial size
        s = torch.cat((m, skip), dim=1)  # self.blockがこれのチャネル数を保証してくれる
        y = self.block(s)
        return y

    def calculate_output_size(self, input_size, skip_size):
        m = self.pool.calculate_output_size(input_size)
        b, c, *s = m
        bs, cs, *ss = skip_size
        assert_eq(s, ss)
        assert_eq(b, bs)
        m = (b, c + cs, *s)
        y = self.block.calculate_output_size(m)
        return y

    @abstractmethod
    def _build_pool(self) -> UpSampling: ...


class UNetHead(BaseUNetModule):
    def __init__(self, input_channel):
        super().__init__()
        self.input_channel = input_channel
        self.bound_assertion(AssertChannel(self.input_channel))

    @abstractmethod
    @classmethod
    def attach_to_unet(cls, unet: "UNet", *args, **kwargs):
        """Dynamically change UNet head."""
        ...

    @abstractmethod
    def calculate_output_size(self, input_size):
        ...

class UNetStem(Block):
    pass


class UNetEncoder(BaseUNetModule):
    def __init__(
        self,
        n_stages: int,
        input_channel: int,
        skip_channels: tuple[int, ...],  # deepest is bottleneck
        pool_strides: tuple[tuple[int, ...], ...],  # deepest is bottleneck
    ):
        super().__init__()
        self.input_channel = input_channel
        self.n_stages = n_stages
        self.pool_strides = pool_strides
        self.skip_channels = skip_channels
        assert_eq(self.n_stages, len(self.pool_strides))
        assert_eq(self.n_stages + 1, len(self.skip_channels))  # stem + stages
        self.stem_channel = self.skip_channels[0]
        self.stem = self._build_stem()
        self.stem.bound_assertion(AssertChannel(self.input_channel, self.stem_channel))
        self.stages = self._build_stages()  # deepest is bottleneck
        assert_eq(self.n_stages, len(self.stages))
        for i in range(self.n_stages):
            self.stages[i].bound_assertion(
                Assertions(
                    AssertChannel(self.skip_channels[i], self.skip_channels[i + 1]),
                    AssertSize(
                        shape_fn=scale_shape_fn(scale=self.pool_strides[i], shrink=True)
                    ),
                )
            )

    @abstractmethod
    def _build_stem(self) -> UNetStem: ...

    @abstractmethod
    def _build_stages(self) -> nn.ModuleList: ...

    def _forward(self, x: Tensor) -> tuple[Tensor, ...]:
        """forward

        Return
        ------

        out: List[Tensor]
            a list of tensor which out[0] is stem feature map and out[-1] is the bottleneck feature map
        """
        hi = self.stem(x)  # (B, C, H, W, D)
        ret = [hi]
        for stage in self.stages:
            stage: EncoderStage
            hi = stage(hi)
            ret.append(hi)
        return tuple(ret)

    def calculate_output_size(self, input_size):
        former_size = self.stem.calculate_output_size(input_size)
        ret = [former_size]
        for i in range(self.n_stages):
            former_size = self.stages[i].calculate_output_size(former_size)
            ret.append(former_size)
        return tuple(ret)


class UNetDecoder(BaseUNetModule):
    def __init__(
        self,
        n_stages: int,
        skip_channels: tuple[int, ...],  # deepest is bottleneck
        pool_strides: tuple[tuple[int, ...]],  # deepest is bottleneck
        *,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.n_stages = n_stages
        self.skip_channels = skip_channels
        self.pool_strides = pool_strides
        self.deep_supervision = deep_supervision
        assert_eq(self.n_stages + 1, len(self.skip_channels))
        assert_eq(self.n_stages, len(self.pool_strides))
        self.head = self._build_head()  # deepest is bottleneck
        self.stages = self._build_stages()  # deepest is bottleneck
        assert_eq(self.n_stages, len(self.stages))
        for i in range(self.n_stages):
            self.stages[i].bound_assertion(
                Assertions(
                    AssertChannel(self.skip_channels[i + 1], self.skip_channels[i]),
                    AssertSize(shape_fn=scale_shape_fn(scale=self.pool_strides[i])),
                )
            )
        if self.deep_supervision:
            assert_eq(self.n_stages + 1, len(self.head))
            for i in range(self.n_stages + 1):
                self.head[i].bound_assertion(AssertChannel(self.skip_channels[i]))
        else:
            self.head.bound_assertion(AssertChannel(self.skip_channels[0]))

    @abstractmethod
    def _build_head(self) -> UNetHead | nn.ModuleList: ...

    @abstractmethod
    def _build_stages(self) -> nn.ModuleList: ...

    def _forward(self, x: tuple[Tensor, ...]) -> Tensor | tuple[Tensor, ...]:
        """forward

        Parameters
        ----------

        x: List[Tensor]
            the feature map from skip connection. x[0] is stem feature map and x[-1] is bottleneck featuremap
            len(x) == n_stages + 1

        Return
        ------
        out: Union[Tensor, List[Tensor]]
        if not self.deep_supervision then Tensor else List[Tensor]
        if List[Tensor], out[-1] is the bottleneck output
        """
        assert_eq(
            self.n_stages + 1,
            len(x),
            f"decoder input starts from stem and its length must be n_stages + 1",
        )
        lo = x[-1]
        r = x[:-1][::-1]
        stages = self.stages[::-1]
        assert_eq(
            len(r),
            len(stages),
            f"decoder input starts from stem and its length must be n_stages + 1",
        )
        ret = [lo]
        for i in range(len(stages)):
            stage: DecoderStage = stages[i]
            lo = stage(lo, r[i])
            ret.append(lo)
        ret = ret[::-1]
        if not self.deep_supervision:
            ret = ret[0]
            h = self.head(ret)
            return h
        assert_eq(len(self.head), len(ret))
        ret = tuple(self.head[i](ret[i]) for i in range(len(self.head)))
        return ret  # deepest is bottleneck

    def calculate_output_size(self, input_size: tuple[tuple[int, ...], ...]):
        assert_eq(
            self.n_stages + 1,
            len(input_size),
            f"decoder input starts from stem and its length must be n_stages + 1",
        )
        lo = input_size[-1]
        r = input_size[:-1][::-1]
        stages = self.stages[::-1]
        assert_eq(
            len(r),
            len(stages),
            f"decoder input starts from stem and its length must be n_stages + 1",
        )
        ret = [lo]
        for i in range(len(stages)):
            stage: DecoderStage = stages[i]
            lo = stage.calculate_output_size(lo, r[i])
            ret.append(lo)
        ret = ret[::-1]
        if not self.deep_supervision:
            ret = ret[0]
            h = self.head.calculate_output_size(ret)
            return h
        assert_eq(len(self.head), len(ret))
        ret = tuple(
            self.head[i].calculate_output_size(ret[i]) for i in range(len(self.head))
        )
        return ret


class UNet(BaseUNetModule):
    def __init__(
        self,
        n_stages: int,
        input_channel: int,
        skip_channels: tuple[int, ...],  # deepest is bottleneck
        pool_strides: (
            int | tuple[int, ...] | list[int] | list[tuple[int, ...]]
        ) = 2,  # deepest is bottleneck,
        *,
        deep_supervision: bool = False,
        dim: int,
    ):
        super().__init__()
        self.n_stages = n_stages
        self.input_channel = input_channel
        """len(skip_channels) == self.n_stages + 1"""
        self.skip_channels = skip_channels
        """len(pool_strides) == self.n_stages"""
        self.pool_strides = pool_strides
        self.deep_supervision = deep_supervision
        self.dim = dim

        self.stem_channel = self.skip_channels[0]

        assert_eq(self.n_stages + 1, len(self.skip_channels))

        if isinstance(self.pool_strides, int):
            self.pool_strides = repeat(self.pool_strides, self.dim)
        if isinstance(self.pool_strides, tuple):
            self.pool_strides = repeat(self.pool_strides, self.n_stages, wrap_type=list)
        elif isinstance(self.pool_strides, list) and isinstance(
            self.pool_strides[0], int
        ):
            self.pool_strides = [repeat(ps, self.dim) for ps in self.pool_strides]

        assert_eq(self.n_stages, len(self.pool_strides))
        self.pool_strides = tuple(self.pool_strides)

        self.encoder = self._build_encoder()
        self.encoder.bound_assertion(
            AssertChannel(input_channel),
        )
        self.decoder = self._build_decoder()

    @abstractmethod
    def _build_encoder(self) -> UNetEncoder: ...

    @abstractmethod
    def _build_decoder(self) -> UNetDecoder: ...

    def _forward(self, x):
        e = self.encoder(x)
        y = self.decoder(e)
        return y

    def calculate_output_size(self, input_size):
        m = self.encoder.calculate_output_size(input_size)
        d = self.decoder.calculate_output_size(m)
        return d

    @staticmethod
    def _plan_to_arguments(
        plan: Plan,
        input_channel,
        skip_channel_ratio: int | tuple = 2,
        deep_supervision: bool = False,
        **kwargs,
    ):
        former = plan.stem_channel
        skip_channels = [former]
        skip_channel_ratio = repeat(skip_channel_ratio, dim=plan.n_stages, types=int)
        for r in skip_channel_ratio:
            former *= r
            skip_channels.append(former)

        unet_arguments = {
            "n_stages": plan.n_stages,
            "input_channel": input_channel,
            "skip_channels": skip_channels,
            "pool_strides": plan.pool_strides,
            "dim": plan.dim,
            "deep_supervision": deep_supervision,
        }
        unet_arguments.update(kwargs)
        return unet_arguments

    @classmethod
    def from_plan(
        cls,
        plan: Plan,
        input_channel,
        skip_channel_ratio: int | tuple = 2,
        deep_supervision: bool = False,
        **kwargs,
    ):
        return cls(
            **cls._plan_to_arguments(
                plan, input_channel, skip_channel_ratio, deep_supervision, **kwargs
            )
        )
