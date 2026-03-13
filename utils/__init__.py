from datetime import datetime
from pathlib import Path

def nowstring():
    return datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

def resolved_path(p: str):
    return Path(p).expanduser().resolve()