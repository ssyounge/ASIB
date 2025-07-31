# utils/progress.py

# tqdm 표시 기본 ON (tty에서만) ➜ PROGRESS=0 으로 끄기
from tqdm import tqdm
import sys
import os

__all__ = ["smart_tqdm"]


def smart_tqdm(iterable, desc=None, **kwargs):
    """A thin wrapper around :class:`tqdm.tqdm`.

    Parameters
    ----------
    iterable : iterable
        The iterable to wrap.
    desc : str, optional
        Description for the progress bar.
    **kwargs : Any
        Additional ``tqdm`` keyword arguments.

    Returns
    -------
    iterator
        ``tqdm`` iterator with sensible defaults.
    """
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("leave", False)
    if "disable" not in kwargs:
        # 1) 환경변수 PROGRESS=0 → 무조건 OFF
        env_off = os.getenv("PROGRESS", "1") == "0"
        kwargs["disable"] = env_off or not sys.stdout.isatty()
    return tqdm(iterable, desc=desc, **kwargs)
