"""Source-tree import shim.

When running from the repository root, this outer folder can shadow the
installable package directory. Re-export the real package so local imports like
`from spark_lshbloom import SparkLSHBloom` still work before installation.
"""

from .spark_lshbloom import (
    LSHBloomConfig,
    SparkLSHBloom,
    detect_suspect_keys,
    duplicate_pairs_to_groups_local,
    emit_band_rows,
    recover_candidate_pairs,
)

__all__ = [
    "LSHBloomConfig",
    "SparkLSHBloom",
    "detect_suspect_keys",
    "duplicate_pairs_to_groups_local",
    "emit_band_rows",
    "recover_candidate_pairs",
]
