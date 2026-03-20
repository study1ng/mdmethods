from abc import ABC, abstractmethod
from torch import Tensor, nn
from experiments.utils import (
    assert_eq,
    assert_divisable,
    assert_shape,
    assert_no_size_change,
    element_wise2,
    prolong,
    scale_shape_fn,
    assert_channel,
    assert_size,
)
import experiments.config
import torch


class Assertions:
    def __init__(self, *assertions):
        self.assertions = assertions

    def __call__(self, *args):
        if experiments.config.assertion:
            for assertion in self.assertions:
                assertion(*args)


class BaseUNetModule(nn.Module, ABC):
    """A marker class for this module's classes"""

    def __init__(self):
        super().__init__()
        self.assertions = None

    def bound_assertion(self, assertions: Assertions):
        if self.assertions is not None:
            self.assertions = Assertions(self.assertions, assertions)
        else:
            self.assertions = assertions

    def forward(self, x):
        y = self._forward(x)
        self._assertions(x, y)
        return y

    @abstractmethod
    def _forward(self, x): ...

    @abstractmethod
    def calculate_output_size(
        self, *input_size: tuple[int, ...] | tuple[tuple[int, ...], ...]
    ) -> tuple[int, ...] | tuple[tuple[int, ...], ...]: ...

    def _assertions(self, x, y):
        if self.assertions is not None:
            self.assertions(x, y)


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
                assert_no_size_change,
                assert_shape(self.input_channel, self.output_channel, dim=1),
            )
        )

    def calculate_output_size(self, input_size):
        b, c, *s = input_size
        return (b, self.output_channel, *s)

class Pool(BaseUNetModule):
    def __init__(
        self, input_channel: int, output_channel, pool_stride,
    ):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.pool_stride = pool_stride
        self.bound_assertion(
            Assertions(
                assert_channel(self.input_channel, self.output_channel),
            )
        )

class DownSampling(Pool):
    def __init__(self, input_channel, output_channel, pool_stride):
        super().__init__(input_channel, output_channel, pool_stride)
        self.bound_assertion(
            Assertions(
                assert_size(shape_fn=scale_shape_fn(scale=self.pool_stride, shrink=True))
            )
        )

    def calculate_output_size(self, input_size):
        b, c, *s = input_size
        assert_eq(self.input_channel, c)
        s = assert_divisable(s, self.pool_stride)
        return (b, self.output_channel, *s)

class UpSampling(Pool):
    def __init__(self, input_channel, output_channel, pool_stride):
        super().__init__(input_channel, output_channel, pool_stride)
        self.bound_assertion(
            Assertions(
                assert_size(shape_fn=scale_shape_fn(scale=self.pool_stride))
            )
        )

    @staticmethod
    @element_wise2(int)
    def _mul(x, y):
        return x * y

    def calculate_output_size(self, input_size):
        b, c, *s = input_size
        assert_eq(self.input_channel, c)
        s = self._mul(s, self.pool_stride)
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
            Assertions(
                assert_channel(self.input_channel, self.pool_channel),
            )
        )
        self.block = self._build_block()
        self.block.bound_assertion(
            Assertions(
                assert_channel(output_shape=self.output_channel),
            )
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
            Assertions(
                assert_size(shape_fn=scale_shape_fn(self.pool_stride, shrink=True))
            )
        )

    @abstractmethod
    def _build_pool(self) -> DownSampling: ...

class DecoderStage(Stage):
    def __init__(self, input_channel, pool_channel, output_channel, pool_stride):
        super().__init__(input_channel, pool_channel, output_channel, pool_stride)
        self.skip_channel = self.output_channel - self.pool_channel
        self.pool.bound_assertion(
            Assertions(assert_size(shape_fn=scale_shape_fn(self.pool_stride)))
        )

    def _forward(self, x, skip):
        m = self.pool(x)
        assert_no_size_change(m, skip)  # same spatial size
        s = torch.cat((m, skip), dim=1)  # self.blockがこれのチャネル数を保証してくれる
        y = self.block(s)
        return y

    def forward(self, x, skip):
        y = self._forward(x, skip)
        self._assertions(x, y)
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
        self.bound_assertion(Assertions(assert_channel(self.input_channel)))


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
        self.stem.bound_assertion(
            Assertions(assert_channel(self.input_channel, self.stem_channel))
        )
        self.stages = self._build_stages()  # deepest is bottleneck
        assert_eq(self.n_stages, len(self.stages))
        for i in range(self.n_stages):
            self.stages[i].bound_assertion(
                Assertions(
                    assert_channel(self.skip_channels[i], self.skip_channels[i + 1]),
                    assert_size(
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
                    assert_channel(self.skip_channels[i + 1], self.skip_channels[i]),
                    assert_size(shape_fn=scale_shape_fn(scale=self.pool_strides[i])),
                )
            )
        if self.deep_supervision:
            assert_eq(self.n_stages + 1, len(self.head))
            for i in range(self.n_stages + 1):
                self.head[i].bound_assertion(
                    Assertions(assert_channel(self.skip_channels[i]))
                )
        else:
            self.head.bound_assertion(Assertions(assert_channel(self.skip_channels[0])))

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
        return ret # deepest is bottleneck

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
        dim: int | None = None,
    ):
        super().__init__()
        self.n_stages = n_stages
        self.input_channel = input_channel
        self.skip_channels = skip_channels
        self.pool_strides = pool_strides
        self.deep_supervision = deep_supervision
        self.dim = dim

        self.stem_channel = self.skip_channels[0]

        assert_eq(self.n_stages + 1, len(self.skip_channels))

        if (
            isinstance(self.pool_strides, int)
            or (
                isinstance(self.pool_strides, list)
                and isinstance(self.pool_strides[0], int)
            )
        ) and self.dim is None:
            raise ValueError(
                "pool_strides is needed to be prolonged but no dim was assigned"
            )

        if isinstance(self.pool_strides, int):
            self.pool_strides: tuple = prolong(self.pool_strides, self.dim, int)

        if isinstance(self.pool_strides, tuple):
            self.pool_strides = prolong(self.pool_strides, self.n_stages - 1, list)
            self.pool_strides = [
                prolong(1, self.dim, int)
            ] + self.pool_strides  # stem doesn't need stride
        elif isinstance(self.pool_strides, list) and isinstance(
            self.pool_strides[0], int
        ):
            self.pool_strides = [prolong(ps, self.dim, int) for ps in self.pool_strides]

        assert_eq(self.n_stages, len(self.pool_strides))
        self.pool_strides = tuple(self.pool_strides)

        self.encoder = self._build_encoder()
        self.encoder.bound_assertion(
            Assertions(
                assert_channel(input_channel),
            )
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
