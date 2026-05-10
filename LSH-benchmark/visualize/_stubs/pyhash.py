from __future__ import annotations


def hash(values) -> int:  # pragma: no cover
    try:
        return int(sum(int(v) for v in values))
    except TypeError:
        return int(values)

