from pathlib import Path
import json
from datetime import datetime


def filekey(filepath: Path | str) -> Path:
    filepath = Path(filepath).name.split(".")[0].split("_")[0]
    return filepath

def resolved_path(p: Path | str) -> Path:
    return Path(p).expanduser().resolve()

def loaded_json(p: Path | str) -> object:
    with resolved_path(p).open("r") as f:
        ret = json.load(f)
    return ret

def nowstring():
    return datetime.now().strftime("%Y.%m.%d.%H.%M.%S")