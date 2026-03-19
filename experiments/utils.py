from pathlib import Path
import json
from datetime import datetime
from shutil import rmtree
import uuid
import threading
from typing import Tuple, Union
from torch import Tensor

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

def divmod_accept_tuple(lhs: int | Tuple[int, ...], rhs: int | Tuple[int, ...]) -> Tuple[Union[int, Tuple[int, ...]], bool]:
    if isinstance(lhs, tuple) and isinstance(rhs, int):
        rhs = (rhs, ) * len(lhs)
    elif isinstance(lhs, int) and isinstance(rhs, tuple):
        lhs = (lhs, ) * len(rhs)
    elif isinstance(lhs, int) and isinstance(rhs, int):
        return lhs // rhs, lhs % rhs == 0
    modulo = (l % r == 0 for l, r in zip(lhs, rhs, strict=True))
    div = (l // r for l, r in zip(lhs, rhs))
    return (div, all(modulo))

def assert_divisable(lhs: int | Tuple[int, ...], rhs: int | Tuple[int, ...]) -> Union[int, Tuple[int, ...]]:
    div, ok = divmod_accept_tuple(lhs, rhs)
    assert ok, f"not dividable, lhs: {lhs}, rhs: {rhs}"
    return div
