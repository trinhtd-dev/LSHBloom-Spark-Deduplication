from __future__ import annotations

from dataclasses import asdict, dataclass

MODE_ALIASES = {
    "bloom": "lshbloom",
    "bloom_filter": "lshbloom",
    "banding": "minhash_lsh",
}


@dataclass(frozen=True)
class LSHBloomConfig:
    threshold: float = 0.85
    num_perm: int = 256
    num_bands: int | None = None
    fp_prob: float = 0.001
    shingle_size: int = 5
    mode: str = "lshbloom"
    expected_items: int = 1_000_000
    seed: int = 42
    deduplicate_chunks: int = 16
    max_shingles: int | None = None

    def __post_init__(self):
        normalized_mode = MODE_ALIASES.get(self.mode, self.mode)
        object.__setattr__(self, "mode", normalized_mode)
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1].")
        if self.num_perm < 2:
            raise ValueError("num_perm must be >= 2.")
        if self.num_bands is not None and self.num_bands <= 0:
            raise ValueError("num_bands must be positive.")
        if not 0.0 < self.fp_prob < 1.0:
            raise ValueError("fp_prob must be in (0, 1).")
        if self.shingle_size <= 0:
            raise ValueError("shingle_size must be positive.")
        if self.mode not in {"lshbloom", "minhash_lsh"}:
            raise ValueError("mode must be 'lshbloom' or 'minhash_lsh'. Aliases: 'bloom', 'bloom_filter', 'banding'.")
        if self.expected_items <= 0:
            raise ValueError("expected_items must be positive.")
        if self.deduplicate_chunks <= 0:
            raise ValueError("deduplicate_chunks must be positive.")
        if self.max_shingles is not None and self.max_shingles <= 0:
            raise ValueError("max_shingles must be positive when set.")
        bands = self.resolved_num_bands
        if self.num_perm % bands != 0:
            raise ValueError("num_perm must be divisible by num_bands.")

    @property
    def resolved_num_bands(self) -> int:
        if self.num_bands is not None:
            return int(self.num_bands)
        return max(1, self.num_perm // 4)

    @property
    def rows_per_band(self) -> int:
        return self.num_perm // self.resolved_num_bands

    def to_dict(self) -> dict:
        data = asdict(self)
        data["num_bands"] = self.resolved_num_bands
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "LSHBloomConfig":
        return cls(**data)
