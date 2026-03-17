from pathlib import Path
import json
from datetime import datetime
from shutil import rmtree


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

def ensure_dir_new(p: Path | str) -> Path:
    p = resolved_path(p)
    p.exists() and rmtree(p)
    p.mkdir(exist_ok=True, parents=True)

def set_symln(from_: Path | str, to: Path | str):
    from_ = resolved_path(from_)
    assert_file_exist(from_)
    to = resolved_path(to)
    to.unlink(missing_ok = True)
    to.symlink_to(from_)

def assert_file_exist(p: Path | str) -> Path:
    assert resolved_path(p).exists(), f"{p} is not exists"