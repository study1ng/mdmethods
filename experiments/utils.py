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


def filekey(filepath: Path | str) -> str:
    """get a id of a path

    Parameters
    ----------
    filepath : Path | str
        the path

    Returns
    -------
    str
        the key
    """
    key = Path(filepath).name.split(".")[0].split("_")[0]
    return key


def resolved_path(p: Path | str) -> Path:
    """return a resolved path

    Parameters
    ----------
    p : Path | str
        path

    Returns
    -------
    Path
        resolved path
    """
    return Path(p).expanduser().resolve()


def loaded_json(p: Path | str) -> object:
    """return a loaded json from path

    Parameters
    ----------
    p : Path | str
        path of json

    Returns
    -------
    object
        json
    """
    assert_file_exist(p)
    with resolved_path(p).open("r") as f:
        ret = json.load(f)
    return ret


def nowstring() -> str:
    """current time string

    Returns
    -------
    str
        current time
    """
    return datetime.now().strftime("%Y.%m.%d.%H.%M.%S")


def repeat(
    v, dim: int, *, types: type | tuple[type, ...] | None = None, wrap_type=tuple
):
    """if v is types, then repeat it for dim times

    Parameters
    ----------
    v : any
        any value to be repeated
    dim : int
        repeat times
    types : type | tuple[type, ...] | None
        target types if None then always wrap
    wrap_type : Function which get a Iterable, optional
        the wrapper, by default tuple

    Returns
    -------
    wrap_type
        the wrapped value

    Notes
    -----
    This function don't assert len(returns) == dim, if v is not instance of types
    """
    if types is None:
        types = type(v)
    if not isinstance(v, types):
        return v
    return wrap_type(v for _ in range(dim))


def element_wise(types = object, wrap_type=tuple):
    """wrapper to make a function can be used for Iterable

    Parameters
    ----------
    types : type | tuple[type, ...]
        the target type
    wrap_type : function which get a Iterable, optional
        the wrapper of return, by default tuple

    Examples
    --------
    ```python
    @element_wise(int)
    def pr(a):
        print(a)
    pr(1)
    >>> 1
    pr((1, 2))
    >>> 1
    >>> 2
    pr("abc") # due to str is not int, str is handled as a iterable and each character enter pr
    >>> a
    >>> b
    >>> c
    ```
    """

    def _element_wise(func):
        @wraps(func)
        def wrapper(arg):
            if isinstance(arg, types):
                return func(arg)
            else:
                return wrap_type(func(a) for a in arg)

        return wrapper

    return _element_wise


def element_wise2(types = object, wrap_type=tuple):
    """like element_wise, but get two argument. if both is not types, assert its length is same

    Parameters
    ----------
    types : type | tuple[type, ...]
        the target type
    wrap_type : function which get a Iterable, optional
        the wrapper of return, by default tuple

    Examples
    --------
    ```python
    @element_wise2(int)
    def add(a, b):
        return a + b
    assert add(1, 2) == 3
    assert add((1, 2), 3) == (4, 5)
    assert add(3, (1, 2)) == (4, 5)
    assert add((1, 2), (3, 4)) == (4, 6)
    assert add("ab", "de") == ("ad", "be") # str is not int so it will be handled as a Iterable and each element character enter add
    """

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
    rmtree(path)


def ensure_dir_new(p: Path | str) -> Path:
    """ensure a directory is empty

    Parameters
    ----------
    p : Path | str
        the target

    Returns
    -------
    Path
        the empty directory
    """
    p = resolved_path(p)
    if p.exists():
        tmp_p = p.with_name(f"{p.name}_trash_{uuid.uuid4().hex}")
        p.rename(tmp_p)
        threading.Thread(target=_bg_rmtree, args=(tmp_p,), daemon=False).start()
    p.mkdir(exist_ok=True, parents=True)
    return p


def set_symln(from_: Path | str, to: Path | str):
    """create a symbolic link from from_ to to

    Parameters
    ----------
    from_ : Path | str
        from
    to : Path | str
        to

    Notes:
    ------
    We assert from_ exist.
    If to exist, it would be removed
    """
    from_ = resolved_path(from_)
    assert_file_exist(from_)
    to = resolved_path(to)
    to.unlink(missing_ok=True)
    to.symlink_to(from_)


def assert_file_exist(p: Path | str):
    """assert a file exist

    Parameters
    ----------
    p : Path | str
        target
    """
    assert resolved_path(p).exists(), f"{p} is not exists"


def assert_eq(expected, found, message: str | None = None):
    """assert two value is existing

    Parameters
    ----------
    expected : any

    found : any

    message : str | None, optional
        any message, by default f"expected {expected}, found {found}"
    """
    assert expected == found, (
        f"expected {expected}, found {found}" if message is None else message
    )


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


def size_of_tensor(x: Tensor) -> Tuple[int, ...]:
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

@element_wise2()
def elementwise_div(l, r):
    return l / r

@element_wise2()
def elementwise_is_divisor(l, r):
    return l % r == 0


@element_wise2()
def elementwise_intdiv(l, r):
    return l // r


@element_wise2()
def elementwise_mul(l, r):
    return l * r

@element_wise2()
def elementwise_gt(l, r):
    return l > r
@element_wise2()
def elementwise_le(l, r):
    return l <= r

@element_wise2()
def elementwise_min(l, r):
    return min(l, r)
@element_wise2()
def elementwise_max(l, r):
    return max(l, r)
def divmod_accept_tuple(
    lhs: int | Tuple[int, ...], rhs: int | Tuple[int, ...]
) -> Tuple[Union[int, Tuple[int, ...]], Union[int, Tuple[int, ...]]]:
    """divmod(lhs, rhs), but accept tuple

    Parameters
    ----------
    lhs : int | Tuple[int, ...]

    rhs : int | Tuple[int, ...]

    Returns
    -------
    Tuple[Union[int, Tuple[int, ...]], Union[int, Tuple[int, ...]]]
        (div, mod)
    """
    return (elementwise_intdiv(lhs, rhs), elementwise_is_divisor(lhs, rhs))


def assert_divisible(
    lhs: int | Tuple[int, ...], rhs: int | Tuple[int, ...]
) -> Union[int, Tuple[int, ...]]:
    """assert rhs is all divisors of lhs, and return lhs // rhs

    Parameters
    ----------
    lhs : int | Tuple[int, ...]

    rhs : int | Tuple[int, ...]


    Returns
    -------
    Union[int, Tuple[int, ...]]
        lhs // rhs
    """
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
    assert_eq(len(gaussian_window_size), len(std))
    return _get_gaussian_kernel(gaussian_window_size, std)


def identity(x):
    return x


def scale_shape_fn(scale: int | tuple[int, ...], shrink=False):
    def _shape_fn(shape: int | tuple[int, ...]):
        if shrink:
            return elementwise_intdiv(shape, scale)
        return elementwise_mul(shape, scale)

    return _shape_fn


size_dim = slice(
    2,
)
channel_dim = 1
