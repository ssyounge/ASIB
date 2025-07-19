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

    # 터미널 X 또는 환경변수 `PROGRESS=0|false` 이면 끈다
    env_off = os.environ.get("PROGRESS", "0").lower() in ("0", "false", "off", "no")

    if "disable" not in kwargs:
        kwargs["disable"] = env_off or (not sys.stdout.isatty())
    return tqdm(iterable, desc=desc, **kwargs)
