from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .hashing import HASH_BACKEND, hash64


class MergeableBloomFilter:
    def __init__(
        self,
        expected_items: int,
        fp_prob: float,
        num_bits: int | None = None,
        num_hashes: int | None = None,
        bits: bytes | bytearray | None = None,
    ):
        self.expected_items = max(1, int(expected_items))
        self.fp_prob = min(max(float(fp_prob), 1e-12), 0.5)
        if num_bits is None:
            num_bits = int(math.ceil(-(self.expected_items * math.log(self.fp_prob)) / (math.log(2) ** 2)))
        if num_hashes is None:
            num_hashes = int(round((num_bits / self.expected_items) * math.log(2)))
        self.num_bits = max(8, int(num_bits))
        self.num_hashes = max(1, int(num_hashes))
        self.num_bytes = (self.num_bits + 7) // 8
        self.bits = bytearray(bits) if bits is not None else bytearray(self.num_bytes)
        if len(self.bits) != self.num_bytes:
            raise ValueError("Bloom bit array size does not match num_bits.")

    def _positions(self, item: str):
        payload = item.encode("utf8", errors="ignore")
        h1 = hash64(payload, seed=0)
        h2 = hash64(payload, seed=1) or 1
        for i in range(self.num_hashes):
            yield (h1 + i * h2) % self.num_bits

    def add(self, item: str) -> None:
        for pos in self._positions(item):
            self.bits[pos >> 3] |= 1 << (pos & 7)

    def __contains__(self, item: str) -> bool:
        for pos in self._positions(item):
            if not (self.bits[pos >> 3] & (1 << (pos & 7))):
                return False
        return True

    def merge(self, other: "MergeableBloomFilter") -> None:
        if self.num_bits != other.num_bits or self.num_hashes != other.num_hashes:
            raise ValueError("Cannot merge Bloom filters with different shapes.")
        a = np.frombuffer(bytes(self.bits), dtype=np.uint8)
        b = np.frombuffer(bytes(other.bits), dtype=np.uint8)
        self.bits = bytearray(np.bitwise_or(a, b).tobytes())

    def to_bytes(self) -> bytes:
        return bytes(self.bits)

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        expected_items: int,
        fp_prob: float,
        num_bits: int,
        num_hashes: int,
    ) -> "MergeableBloomFilter":
        return cls(
            expected_items=expected_items,
            fp_prob=fp_prob,
            num_bits=num_bits,
            num_hashes=num_hashes,
            bits=payload,
        )

    def metadata(self) -> dict:
        return {
            "expected_items": self.expected_items,
            "fp_prob": self.fp_prob,
            "num_bits": self.num_bits,
            "num_hashes": self.num_hashes,
            "num_bytes": self.num_bytes,
            "hash_backend": HASH_BACKEND,
        }

    def save(self, path: str | Path) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        (root / "bloom.bin").write_bytes(self.to_bytes())
        (root / "bloom.json").write_text(json.dumps(self.metadata(), indent=2), encoding="utf8")

    @classmethod
    def load(cls, path: str | Path) -> "MergeableBloomFilter":
        root = Path(path)
        meta = json.loads((root / "bloom.json").read_text(encoding="utf8"))
        saved_backend = meta.get("hash_backend")
        if saved_backend is not None and saved_backend != HASH_BACKEND:
            raise RuntimeError(
                f"Bloom filter was built with hash backend '{saved_backend}' but the current "
                f"runtime uses '{HASH_BACKEND}'. Install/uninstall xxhash to match, otherwise "
                f"all queries will silently miss."
            )
        payload = (root / "bloom.bin").read_bytes()
        return cls.from_bytes(payload=payload, **{k: meta[k] for k in ("expected_items", "fp_prob", "num_bits", "num_hashes")})
