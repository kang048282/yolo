from functools import wraps
from time import time


def is_video(ext: str):
    """
    Returns true if ext exists in
    allowed_exts for video files.

    Args:
        ext:

    Returns:

    """
    allowed_exts = (".mp4", ".webm", ".ogg", ".avi", ".wmv", ".mkv", ".3gp")
    return any(ext.endswith(x) for x in allowed_exts)


def tik_tok(func):
    """
    Keep track of time for each process.

    Args:
        func:

    Returns:

    """

    @wraps(func)
    def _time_it(*args, **kwargs):
        start = time()
        try:
            return func(*args, **kwargs)
        finally:
            end_ = time()
            print(f"time: {end_ - start:.03f}s, fps: {1 / (end_ - start):.03f}")

    return _time_it
