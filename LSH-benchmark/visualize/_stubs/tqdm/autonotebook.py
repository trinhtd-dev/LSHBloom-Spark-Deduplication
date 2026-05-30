from __future__ import annotations


class _Tqdm:
    def __init__(self, total=None, desc=None):
        self.total = total
        self.desc = desc
        self.n = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, n=1):
        self.n += n

    def close(self):
        return None


def tqdm(*args, **kwargs):
    return _Tqdm(*args, **kwargs)

