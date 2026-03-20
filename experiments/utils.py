from pathlib import Path
import json
from datetime import datetime
from functools import wraps
from shutil import rmtree
import uuid
import threading
from typing import Tuple, Union
from torch import Tensor
import torch

def filekey(filepath: Path | str) -> Path:
    filepath = Path(filepath).name.split(".")[0].split("_")[0]
    return filepath

def resolved_path(p: Path | str) -> Path:
    return Path(p).expanduser().resolve()

def loaded_json(p: Path | str) -> object:
    assert_file_exist(p)
    with resolved_path(p).open("r") as f:
        ret = json.load(f)
    return ret

def nowstring():
    return datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

def prolong(v, dim: int, types, wrap_type=tuple):
    if not isinstance(v, types):
        return v
    return wrap_type(v for _ in range(dim))

def element_wise(types, wrap_type = tuple):
    def _element_wise(func):
        @wraps(func)
        def wrapper(arg):
            if isinstance(arg, types):
                return func(arg)
            else:
                return wrap_type(func(a) for a in arg)
        return wrapper
    return _element_wise

def element_wise2(types, wrap_type = tuple):
    def _element_wise2(func):
        @wraps(func)
        def wrapper(arg1, arg2):
            if isinstance(arg1, types) and isinstance(arg2, types):
                return func(arg1, arg2)
            if isinstance(arg1, types) and not isinstance(arg2, types):
                return wrap_type(func(arg1, a) for a in arg2)
            if not isinstance(arg1, types) and isinstance(arg2, types):
                return wrap_type(func(a, arg2) for a in arg1)
            return wrap_type(func(a, b) for a, b in zip(arg1, arg2, strict=True))
        return wrapper
    return _element_wise2

def _bg_rmtree(path: Path):
    """バックグラウンドで安全に削除を実行する"""
    rmtree(path)
def ensure_dir_new(p: Path | str) -> Path:
    p = resolved_path(p)
    if p.exists():
        tmp_p = p.with_name(f"{p.name}_trash_{uuid.uuid4().hex}")
        p.rename(tmp_p)
        threading.Thread(target=_bg_rmtree, args=(tmp_p,), daemon=False).start()
    p.mkdir(exist_ok=True, parents=True)
    return p

def set_symln(from_: Path | str, to: Path | str):
    from_ = resolved_path(from_)
    assert_file_exist(from_)
    to = resolved_path(to)
    to.unlink(missing_ok = True)
    to.symlink_to(from_)

def assert_file_exist(p: Path | str) -> Path:
    assert resolved_path(p).exists(), f"{p} is not exists"

def assert_eq(expected, found, message: str | None = None):
    assert expected == found, f"expected {expected}, found {found}" if message is None else message


def channel_of_tensor(x: Tensor) -> int:
    return x.shape[1]

def size_of_tensor(x: Tensor) -> Tuple[int, ...]:
    return x.shape[2:]

@element_wise2(int)
def _modulo(l, r):
    return l % r == 0

@element_wise2(int)
def _div(l, r):
    return l // r

def divmod_accept_tuple(lhs: int | Tuple[int, ...], rhs: int | Tuple[int, ...]) -> Tuple[Union[int, Tuple[int, ...]], Union[int, Tuple[int, ...]]]:
    return (_div(lhs, rhs), _modulo(lhs, rhs))

def assert_divisable(lhs: int | Tuple[int, ...], rhs: int | Tuple[int, ...]) -> Union[int, Tuple[int, ...]]:
    div, m = divmod_accept_tuple(lhs, rhs)
    assert all(m), f"not dividable, lhs: {lhs}, rhs: {rhs}"
    return div

def _get_gaussian_kernel(gaussian_window_size: tuple[int, ...], std: tuple[float, ...]):
    assert_eq(gaussian_window_size, std)
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

def get_gaussian_kernel(gaussian_window_size: int | tuple[int, ...], std: float | tuple[float], dim: int | None = None):
    if dim is None:
        if isinstance(gaussian_window_size, (tuple, list)):
            dim = len(gaussian_window_size)
        elif isinstance(std, (tuple, list)):
            dim = len(std)
        else:
            dim = 1 # 1d
    if isinstance(gaussian_window_size, int):
        gaussian_window_size = (gaussian_window_size,) * dim
    if isinstance(std, (int, float)):
        std = (std,) * dim
    assert_eq(len(gaussian_window_size), len(std))
    return _get_gaussian_kernel(gaussian_window_size, std)