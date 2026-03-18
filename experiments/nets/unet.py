from torch import nn
from typing import List, Union, Tuple
from abc import ABC, abstractmethod, abstractclassmethod,

class UNet(nn.Module, ABC):
    def __init__(
        self,
        input_size: Tuple[int, ...],
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
    ):
        nn.Module.__init__()
        self.input_size = input_size
        self.input_channels = input_channels
        self.n_stages = n_stages

        if isinstance(features_per_stage, int):
            self.output_channels = [features_per_stage] * n_stages
        else:
            self.output_channels = list(features_per_stage)

        spatial_dims = len(input_size)
        _strides = []
        
        if isinstance(strides, int):
            _strides = [[strides] * spatial_dims for _ in range(n_stages)]
        elif isinstance(strides, (list, tuple)):
            for s in strides:
                if isinstance(s, int):
                    _strides.append([s] * spatial_dims)
                elif isinstance(s, (list, tuple)):
                    _strides.append(list(s))

        self.output_size = []
        current_spatial_size = list(input_size)

        for i in range(n_stages):
            current_spatial_size = [
                size // stride for size, stride in zip(current_spatial_size, _strides[i])
            ]
            self.output_size.append(tuple(current_spatial_size))

class UNetEncoder(nn.Module):
