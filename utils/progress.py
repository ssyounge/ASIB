# utils/progress.py

# tqdm 표시 기본 OFF ➜ PROGRESS=1 로 켜기
from tqdm import tqdm
import sys, os

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
    # PROGRESS=1 을 주면 강제 활성화, 기본은 OFF
    env_flag = os.environ.get("PROGRESS", "0").lower() in ("1", "true", "yes", "on")
    kwargs["disable"] = not env_flag
    return tqdm(iterable, desc=desc, **kwargs)
