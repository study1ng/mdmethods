from datetime import datetime
from torch import Tensor
from experiments.utils.fsutils import *
from experiments.utils.wraputils import *
import torch

def nowstring() -> str:
    """current time string

    Returns
    -------
    str
        current time
    """
    return datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

def channel_of_tensor(x: Tensor) -> int:
    """sugar syntax for x.shape[1]

    Parameters
    ----------
    x : Tensor
        expect (B,C,H,W,D)

    Returns
    -------
    int
        channel
    """
    return x.shape[1]


def size_of_tensor(x: Tensor) -> tuple[int, ...]:
    """sugar syntax for x.shape[2:]

    Parameters
    ----------
    x : Tensor
        expect (B,C,H,...)

    Returns
    -------
    Tuple[int, ...]
        size
    """
    return x.shape[2:]



def _get_gaussian_kernel(gaussian_window_size: tuple[int, ...], std: tuple[float, ...]):
    AssertEq()(gaussian_window_size, std)

    def _1dkernel(ws: int, s: int):
        n = torch.arange(0, ws).float()
        n -= n.mean()
        n /= s
        w = torch.exp(-0.5 * n**2)
        return w

    _1dkernels = tuple(_1dkernel(ws, s) for ws, s in zip(gaussian_window_size, std))
    former = _1dkernels[0]
    for kernel in _1dkernels[1:]:
        former = former[..., None] * kernel
    former /= former.sum()
    return former


def get_gaussian_kernel(
    gaussian_window_size: int | tuple[int, ...],
    std: float | tuple[float],
    dim: int | None = None,
):
    """return the gaussian kernel

    Parameters
    ----------
    gaussian_window_size : int | tuple[int, ...]
        the gaussian window width, for each dimension
    std : float | tuple[float]
        the standard deviation, for each dimension
    dim : int | None, optional
        dimension size, by default None

    Returns
    -------
    Tensor
        the gaussian kernel
    """
    if dim is None:
        if isinstance(gaussian_window_size, (tuple, list)):
            dim = len(gaussian_window_size)
        elif isinstance(std, (tuple, list)):
            dim = len(std)
        else:
            dim = 1  # 1d
    if isinstance(gaussian_window_size, int):
        gaussian_window_size = (gaussian_window_size,) * dim
    if isinstance(std, (int, float)):
        std = (std,) * dim
    AssertEq()(len(gaussian_window_size), len(std))
    return _get_gaussian_kernel(gaussian_window_size, std)
