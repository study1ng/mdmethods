from torch import nn
from typing import List, Union, Tuple
from abc import ABC, abstractmethod
from torch import Tensor
import torch
from experiments.utils import (
    assert_eq,
    channel_of_tensor,
    size_of_tensor,
    assert_divisable,
)
from experiments.plan import Plan


class Block(nn.Module, ABC):
    """Blockは, ある(B, C_1, H, W, D)の入力を受けて, (B, C_2, H, W, D)の出力を返すような写像として定義する."""

    def __init__(self, input_channel: int, output_channel: int, size: Tuple[int, ...]):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.size = size
        self.dim = len(size)

    def forward(self, x: Tensor):
        assert_eq(self.input_channel, channel_of_tensor(x))
        assert_eq(self.size, size_of_tensor(x))
        ret = self._forward(x)
        assert_eq(self.output_channel, channel_of_tensor(ret))
        assert_eq(
            self.size,
            size_of_tensor(ret),
            f"block size should not be changed during forward",
        )
        return ret

    @abstractmethod
    def _forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------

        x: Tensor

        Return
        ------

        out: Tensor, its shape will be constant during io, out.shape[2:] == x.shape[2:]
        """
        ...


class Stage(nn.Module, ABC):
    def __init__(
        self,
        input_channel: int,
        after_sample_channel: int,
        output_channel: int,
        input_size: tuple[int, ...],
        output_size: tuple[int, ...],
        pool_stride: int | tuple[int, ...] = 2,
    ):
        super().__init__()
        self.input_channel = input_channel
        self.after_sample_channel = after_sample_channel
        self.output_channel = output_channel
        self.input_size = input_size
        self.output_size = output_size
        self.pool_stride = pool_stride
        self.dim = len(self.input_size)
        if isinstance(self.pool_stride, int):
            self.pool_stride = (self.pool_stride,) * self.dim
        self.sample = self._build_sample()
        self.block = self._build_block()

    @abstractmethod
    def _build_sample(self) -> nn.Module: ...

    @abstractmethod
    def _build_block(self) -> Block: ...


class EncoderStage(Stage):
    """
    EncoderStageとは, ダウンサンプリングまたはアップサンプリングを行う関数fとブロックgを用いて, g(f(x))を行う写像
    """
    def __init__(self, input_channel, after_sample_channel, output_channel, input_size, output_size, pool_stride = 2):
        super().__init__(input_channel, after_sample_channel, output_channel, input_size, output_size, pool_stride)
        assert_eq(assert_divisable(self.input_size, self.output_size), self.pool_stride)

    def forward(self, x: Tensor) -> Tensor:
        assert_eq(self.input_channel, channel_of_tensor(x))
        assert_eq(self.input_size, size_of_tensor(x))
        sampled = self.sample(x)
        assert_eq(self.after_sample_channel, channel_of_tensor(sampled))
        assert_eq(self.output_size, size_of_tensor(sampled))
        out = self.block(sampled)
        assert_eq(self.output_channel, channel_of_tensor(out))
        return out


class DecoderStage(Stage):
    """
    DecoderStageとは, アップサンプリングを行う関数fとブロックgを用いて, g(f(x) + skip)を行う写像
    """

    def __init__(
        self,
        input_channel,
        after_sample_channel,
        skip_channel,
        output_channel,
        input_size,
        output_size,
        pool_stride: int | tuple[int, ...] = 2,
    ):
        self.skip_channel = skip_channel
        super().__init__(
            input_channel,
            after_sample_channel,
            output_channel,
            input_size,
            output_size,
            pool_stride=pool_stride,
        )
        assert_eq(assert_divisable(self.output_size, self.input_size), self.pool_stride)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        assert_eq(self.skip_channel, channel_of_tensor(skip))
        assert_eq(self.output_size, size_of_tensor(skip))
        assert_eq(self.input_size, size_of_tensor(x))
        sampled = self.sample(x)
        assert_eq(self.after_sample_channel, channel_of_tensor(sampled))
        skipped = torch.cat((sampled, skip), dim=1)
        out = self.block(skipped)
        assert_eq(self.output_channel, channel_of_tensor(out))
        return out

UNetStem = Block
UNetHead = Block

class UNetEncoder(nn.Module, ABC):
    """UNet Encoder

    this module project the result of stem to the input of decoder
    """

    def __init__(
        self,
        input_size: Tuple[int, ...],
        input_channel: int,
        stem_channel: int,
        n_stages: int,
        skip_channels: List[int],
        skip_size: List[Tuple[int, ...]],
        conv_kernel_size: Union[
            int, List[int], Tuple[int, ...], List[Tuple[int, ...]]
        ] = 3,
        pool_strides: Union[int, List[int], Tuple[int, ...], List[Tuple[int, ...]]] = 2,
        pool_channel_increase_ratio: Union[int, List[int]] = 2,
    ):
        """UNetEncoder

        this module project the result of stem to the input of decoder

        Parameters
        ----------

        input_size : Tuple[int, ...]
            input size, expect like (H, W, D) if 3D
        input_channel: int
            input channel, expect be 1 if CT
        stem_channel: int
            the channel size of stem,
        n_stages: int
            the count of stages , expect like 4
        conv_kernel_size: Union[int, List[int], Tuple[int, ...], List[Tuple[int, ...]]]
            the shape of conv kernel size, expect like [(3,3,3), (3,3,3), (3,3,3), (3,3,3)]. 3, (3,3,3), [3,3,3,3] is same meaning
            default to 3
        pool_strides: Union[int, List[int], Tuple[int, ...], List[Tuple[int, ...]]]
            the stride of pool op. expect like [(1,1,1),(2,2,2),(2,2,2),(2,2,2)]. the first element is recommended to be 1.
            2, [1,2,2,2], (2,2,2) is same meaning. default to 2
        pool_channel_increase_ratio: Union[int, List[int]]
            the channel increase ratio of pool op. expect like [2,2,2,2]. 2 is same meaning. default to 2
        """
        super().__init__()
        self.input_size = input_size
        self.input_channel = input_channel
        self.stem_channel = stem_channel
        self.n_stages = n_stages
        self.conv_kernel_size = conv_kernel_size
        self.skip_channels = skip_channels
        self.skip_size = skip_size
        self.pool_strides = pool_strides
        self.pool_channel_increase_ratio = pool_channel_increase_ratio
        self.stem = self._build_stem()
        self.stages = self._build_stages()
        assert (
            len(self.stages) == n_stages
        ), f"len(self.stages) should be same as n_stages, check build_stages"

    @abstractmethod
    def _build_stem(self) -> UNetStem: ...

    @abstractmethod
    def _build_stages(self) -> nn.ModuleList[EncoderStage]:
        """
        Return
        ------

        out: nn.ModuleList[Stage]
            out[0] is the shallowest stage, out[-1] is the bottle neck stage
        """
        ...

    @abstractmethod
    def forward(self, x: Tensor) -> List[Tensor]:
        """forward

        Return
        ------

        out: List[Tensor]
            a list of tensor which out[0] is stem feature map and out[-1] is the bottleneck feature map
        """
        assert_eq(self.input_size, size_of_tensor(x))
        assert_eq(self.input_channel, channel_of_tensor(x))
        hi = self.stem(x)  # (B, C, H, W, D)
        assert_eq(self.stem_channel, channel_of_tensor(hi))
        ret = [hi]
        for stage in self.stages:
            stage: EncoderStage
            hi = stage(hi)
            ret.append(hi)
        return ret


class UNetDecoder(nn.Module, ABC):
    """
    UNetDecoderは, 長さn_stagesのStageのリストとdeep_supervision時にはBlockのリストを, そうでなければBlockを一つ持つ
    """

    def __init__(
        self,
        output_channel: int,
        skip_size: List[Tuple[int, ...]],
        skip_channels: List[int],
        n_stages: int,
        conv_kernel_size: Union[
            int, List[int], Tuple[int, ...], List[Tuple[int, ...]]
        ] = 3,
        pool_strides: Union[int, List[int], Tuple[int, ...], List[Tuple[int, ...]]] = 2,
        pool_channel_increase_ratio: Union[int, List[int]] = 2,
        deep_supervision: bool = False,
    ):
        """UNetDecoder

        Parameters
        ----------

        output_channel: int
            the class of output
        skip_size: List[Tuple[int, ...]]
            the feature map size of each skip, 0 is stem, -1 is deepest, expected to be like (H, W, D) if 3D
        skip_channels: List[int]
            the feature map channel of each skip, 0 is stem, -1 is deepest,
        n_stages: int
            the count of stages , expect like 4
        conv_kernel_size: Union[int, List[int], Tuple[int, ...], List[Tuple[int, ...]]]
            the shape of conv kernel size, expect like [(3,3,3), (3,3,3), (3,3,3), (3,3,3)]. 3, (3,3,3), [3,3,3,3] is same meaning
            default to 3
        pool_strides: Union[int, List[int], Tuple[int, ...], List[Tuple[int, ...]]]
            the stride of pool op. expect like [(1,1,1),(2,2,2),(2,2,2),(2,2,2)]. the first element is recommended to be 1.
            2, [1,2,2,2], (2,2,2) is same meaning. default to 2
        pool_channel_increase_ratio: Union[int, List[int]]
            the channel increase ratio of pool op. expect like [2,2,2,2]. 2 is same meaning. default to 2
        deep_supervision: bool.
            default to False
        """
        super().__init__()
        self.output_channel = output_channel
        self.skip_size = skip_size
        self.skip_channels = skip_channels
        self.n_stages = n_stages
        assert_eq(
            self.n_stages + 1,
            len(self.skip_channels),
            f"skip channels length {len(self.skip_channels)} must be same as n_stages {self.n_stages} + 1",
        )

        assert_eq(
            self.n_stages + 1,
            len(self.skip_size),
            f"skip size length {len(self.skip_size)} must be same as n_stages {self.n_stages} + 1",
        )
        self.conv_kernel_size = conv_kernel_size
        self.pool_strides = pool_strides
        self.pool_channel_increase_ratio = pool_channel_increase_ratio
        self.deep_supervision = deep_supervision
        self.head = self._build_head()
        self.stages = self._build_stages()

    @abstractmethod
    def _build_stages(self) -> nn.ModuleList[DecoderStage]:
        """
        Return
        ------

        nn.ModuleList[DecoderStage]
        out[0] is the shallowest stage, out[-1] is the bottleneck stage
        """
        ...

    @abstractmethod
    def _build_head(self) -> Union[UNetHead, nn.ModuleList[UNetHead]]:
        """
        Return
        ------
        Union["UNetHead", nn.ModuleList["UNetHead"]]

        out[0] is the shallowest stage, out[-1] is the bottleneck stage if deep_supervision
        """
        ...

    @abstractmethod
    def forward(self, x: List[Tensor]) -> Union[Tensor, List[Tensor]]:
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
            assert_eq(self.output_channel, channel_of_tensor(h))
            return h
        assert_eq(len(self.head), len(ret))
        ret = [self.head[i](ret[i]) for i in range(len(self.head))]
        assert_eq(self.output_channel, channel_of_tensor(ret[0]))
        return ret


class UNet(nn.Module, ABC):
    """UNet

    UNetはencoderとdecoderの二つのモジュールから表されるモデルとして定義する.
    """

    def __init__(
        self,
        patch_size: Tuple[int, ...],
        patch_channel: int,
        stem_channel: int,
        output_channel: int,
        n_stages: int,
        conv_kernel_size: Union[
            int, List[int], Tuple[int, ...], List[Tuple[int, ...]]
        ] = 3,
        pool_strides: Union[int, List[int], Tuple[int, ...], List[Tuple[int, ...]]] = 2,
        pool_channel_increase_ratio: Union[int, List[int]] = 2,
        deep_supervision: bool = False,
    ):
        """UNet

        Parameters
        ----------
        patch_size : Tuple[int, ...]
            patch size, expect like (H, W, D) if 3D
        patch_channel: int
            the channel size of patch, expect 1 if CT
        stem_channel: int
            the channel size of stem,
        output_channel: int
            the channel size of output, expect like its class count + 1(background)
        n_stages: int
            the count of stages , expect like 4
        conv_kernel_size: Union[int, List[int], Tuple[int, ...], List[Tuple[int, ...]]]
            the shape of conv kernel size, expect like [(3,3,3), (3,3,3), (3,3,3), (3,3,3)]. 3, (3,3,3), [3,3,3,3] is same meaning
            default to 3
        pool_strides: Union[int, List[int], Tuple[int, ...], List[Tuple[int, ...]]]
            the stride of pool op. expect like [(1,1,1),(2,2,2),(2,2,2),(2,2,2)]. the first element is recommended to be 1.
            2, [1,2,2,2], (2,2,2) is same meaning. default to 2
        pool_channel_increase_ratio: Union[int, List[int]]
            the channel increase ratio of pool op. expect like [2,2,2,2]. 2 is same meaning. default to 2
        deep_supervision: bool.
            default to False
        """
        super().__init__()
        self.patch_size = patch_size
        self.dim = len(patch_size)
        self.patch_channel = patch_channel
        self.output_channel = output_channel
        self.stem_channel = stem_channel
        self.n_stages = n_stages
        self.conv_kernel_size = conv_kernel_size
        self.pool_strides = pool_strides
        self.pool_channel_increase_ratio = pool_channel_increase_ratio
        self.deep_supervision = deep_supervision

        self.conv_kernel_size = self._prolong(self.conv_kernel_size)
        self.pool_strides = self._prolong(self.pool_strides, length=self.n_stages - 1)
        if len(self.pool_strides) == self.n_stages - 1:
            self.pool_strides = [(1,) * self.dim] + self.pool_strides
        if isinstance(self.pool_channel_increase_ratio, int):
            self.pool_channel_increase_ratio = [
                self.pool_channel_increase_ratio
            ] * self.n_stages

        # スキップ接続の特徴マップの形状を計算する
        self.skip_channels = [self.stem_channel]
        former = self.stem_channel
        for ratio in self.pool_channel_increase_ratio:
            former *= ratio
            self.skip_channels.append(former)

        self.skip_size = [self.patch_size]
        former = self.patch_size
        for stride in self.pool_strides:
            div = tuple(f // s for f, s in zip(former, stride))
            norem = tuple(f % s == 0 for f, s in zip(former, stride))
            assert all(
                norem
            ), f"feature map size {former} is not dividable by kernel size {stride}"
            self.skip_size.append(div)
            former = div

        print(f"skip feature map channels: {self.skip_channels}")
        print(f"skip feature map size: {self.skip_size}")

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _prolong(
        self,
        value: Union[int, List[int], Tuple[int, ...], List[Tuple[int, ...]]],
        length: int | None = None,
        dim: int | None = None,
    ):
        """もしvalueが一つのステージのみに適用される値ならばlength個に複製し, 一つの次元のみに適用される値ならばdim個に複製する"""
        if dim is None:
            dim = self.dim
        if length is None:
            length = self.n_stages
        if isinstance(value, int):
            value = (value,) * dim
        if isinstance(value, tuple):
            return [value] * length
        if isinstance(value, list) and isinstance(value[0], int):
            return [(v,) * dim for v in value]
        return value

    @abstractmethod
    def _build_encoder(self) -> UNetEncoder: ...

    @abstractmethod
    def _build_decoder(self) -> UNetDecoder: ...

    def forward(self, x) -> Union[Tensor, List[Tensor]]:
        e = self.encoder(x)
        assert all(
            es.shape[1:] == (c,) + s
            for es, c, s in zip(e, self.skip_channels, self.skip_size, strict=True)
        ), f"encoder output shape: {[es.shape[1:] for es in e]}, expected skip channel: {self.skip_channels}, expected_skip_size: {self.skip_size}"
        d = self.decoder(e)
        if self.deep_supervision:
            assert_eq(self.n_stages + 1, len(d))
            for i in range(len(d)):
                assert_eq(self.output_channel, channel_of_tensor(d[i]))
                assert_eq(self.skip_size[i], size_of_tensor(d[i]))
            assert_eq(self.patch_size, size_of_tensor(d[0]))
        else:
            assert_eq(self.patch_size, size_of_tensor(d))
            assert_eq(self.output_channel, channel_of_tensor(d))
        return d

    @classmethod
    def from_plan(
        cls,
        plan: Plan,
        patch_channel: int,
        output_channel,
        pool_channel_increase_ratio=2,
        deep_supervision: bool = False,
        **kwargs
    ) -> "UNet":
        return cls(
            patch_size=plan.patch_size,
            patch_channel=patch_channel,
            stem_channel=plan.stem_channel,
            output_channel=output_channel,
            conv_kernel_size=plan.conv_kernel_size,
            pool_strides=plan.pool_strides,
            pool_channel_increase_ratio=pool_channel_increase_ratio,
            deep_supervision=deep_supervision,
            n_stages=len(plan.pool_strides),
            **kwargs
        )
