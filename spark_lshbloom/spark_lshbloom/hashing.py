from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np

try:
    import xxhash
except ImportError:  # pragma: no cover - optional speed dependency
    xxhash = None

MERSENNE_PRIME = (1 << 61) - 1
MAX_HASH = (1 << 64) - 1


def char_ngrams(text: str, n: int, max_shingles: int | None = None) -> list[str]:
    if text is None:
        return []
    value = " ".join(str(text).lower().split())
    if not value:
        return []
    if len(value) <= n:
        return [value]

    seen = set()
    shingles = []
    for i in range(len(value) - n + 1):
        shingle = value[i : i + n]
        if shingle in seen:
            continue
        seen.add(shingle)
        shingles.append(shingle)
        if max_shingles is not None and len(shingles) >= max_shingles:
            break
    return shingles


def hash64(value: str | bytes, seed: int = 0) -> int:
    payload = value if isinstance(value, bytes) else value.encode("utf8", errors="ignore")
    if xxhash is not None:
        return xxhash.xxh64_intdigest(payload, seed=seed & 0xFFFFFFFF)
    person = int(seed).to_bytes(8, "little", signed=False)
    digest = hashlib.blake2b(payload, digest_size=8, person=person).digest()
    return int.from_bytes(digest, "big", signed=False)


@lru_cache(maxsize=32)
def permutation_coefficients(num_perm: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, MERSENNE_PRIME, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, MERSENNE_PRIME, size=num_perm, dtype=np.uint64)
    return a, b


def minhash_signature(
    text: str,
    num_perm: int,
    shingle_size: int,
    seed: int = 42,
    max_shingles: int | None = None,
) -> list[int]:
    shingles = char_ngrams(text, shingle_size, max_shingles=max_shingles)
    if not shingles:
        shingles = ["__spark_lshbloom_empty__"]

    a, b = permutation_coefficients(num_perm, seed)
    signature = np.full(num_perm, MAX_HASH, dtype=np.uint64)
    prime = np.uint64(MERSENNE_PRIME)

    for shingle in shingles:
        x = np.uint64(hash64(shingle, seed=seed) % MERSENNE_PRIME)
        values = (a * x + b) % prime
        signature = np.minimum(signature, values)

    return [int(x) for x in signature]


def band_keys_from_signature(signature: list[int], num_bands: int) -> list[str]:
    if not signature:
        return []
    signature_array = np.asarray(signature, dtype=np.uint64)
    rows_per_band = len(signature) // num_bands
    keys = []
    for band_id in range(num_bands):
        start = band_id * rows_per_band
        end = start + rows_per_band
        payload = signature_array[start:end].byteswap().tobytes()
        keys.append(f"{band_id}:{hash64(payload, seed=band_id):016x}")
    return keys


def band_keys_for_text(
    text: str,
    num_perm: int,
    num_bands: int,
    shingle_size: int,
    seed: int = 42,
    max_shingles: int | None = None,
) -> list[str]:
    signature = minhash_signature(
        text,
        num_perm=num_perm,
        shingle_size=shingle_size,
        seed=seed,
        max_shingles=max_shingles,
    )
    return band_keys_from_signature(signature, num_bands=num_bands)
