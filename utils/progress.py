# utils/progress.py

# tqdm 막대를 전역으로 끄려면  ➜  export TQDM_DISABLE=1
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

    # ── NEW ── 전역 비활성화 플래그 (로그 화면 정리용)
    env_disable = os.getenv("TQDM_DISABLE", "0") == "1"

    if "disable" not in kwargs:
        kwargs["disable"] = env_disable or (not sys.stdout.isatty())
    return tqdm(iterable, desc=desc, **kwargs)
