from __future__ import annotations
import os
import pathlib


def to_writable(path: str | os.PathLike, *, env_var: str = "ASMB_KD_ROOT") -> str:
    """Return a writable absolute path.

    - Absolute paths are returned unchanged.
    - Relative paths are resolved under ``$ASMB_KD_ROOT`` (or ``$HOME/.asmb_kd``).
    The parent directory is created if it does not exist.
    """
    p = pathlib.Path(path)
    if p.is_absolute():
        abs_path = p
    else:
        root = os.getenv(env_var) or (pathlib.Path.home() / ".asmb_kd")
        abs_path = pathlib.Path(root) / p
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    return str(abs_path)
