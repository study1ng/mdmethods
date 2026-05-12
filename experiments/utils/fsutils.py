from pathlib import Path
import json, threading, uuid
from shutil import rmtree

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
