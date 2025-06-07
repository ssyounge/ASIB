from tqdm import tqdm
import sys

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
        kwargs["disable"] = not sys.stdout.isatty()
    return tqdm(iterable, desc=desc, **kwargs)
