from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .bloom import MergeableBloomFilter
from .config import LSHBloomConfig
from .hashing import band_keys_for_text


class SparkLSHBloom:
    def __init__(
        self,
        threshold: float = 0.85,
        num_perm: int = 256,
        num_bands: int | None = None,
        fp_prob: float = 0.001,
        shingle_size: int = 5,
        mode: str = "lshbloom",
        expected_items: int = 1_000_000,
        seed: int = 42,
        deduplicate_chunks: int = 16,
        use_pandas_udf: bool = True,
        max_shingles: int | None = None,
    ):
        self.config = LSHBloomConfig(
            threshold=threshold,
            num_perm=num_perm,
            num_bands=num_bands,
            fp_prob=fp_prob,
            shingle_size=shingle_size,
            mode=mode,
            expected_items=expected_items,
            seed=seed,
            deduplicate_chunks=deduplicate_chunks,
            max_shingles=max_shingles,
        )
        self.bloom: MergeableBloomFilter | None = None
        self.use_pandas_udf = bool(use_pandas_udf)

    @property
    def mode(self) -> str:
        return self.config.mode

    def per_document_fp(self) -> float:
        """Approximate per-document false-positive rate.

        A document is flagged as duplicate if ANY of its `num_bands` band-keys
        hits the Bloom. With per-key FP `p`, the per-doc FP is `1 - (1-p)^b`,
        which is substantially larger than `p` for the typical b in [16, 64].
        """
        p = self.config.fp_prob
        b = self.config.resolved_num_bands
        return 1.0 - (1.0 - p) ** b

    def _make_empty_bloom(self) -> MergeableBloomFilter:
        return MergeableBloomFilter(
            expected_items=self.config.expected_items * self.config.resolved_num_bands,
            fp_prob=self.config.fp_prob,
        )

    def _band_keys_column(self, df, text_col: str, output_col: str = "band_keys"):
        from pyspark.sql.functions import pandas_udf, udf
        from pyspark.sql.types import ArrayType, StringType

        cfg = self.config

        def make_one(text: str) -> list[str]:
            return band_keys_for_text(
                text=str(text) if text is not None else "",
                num_perm=cfg.num_perm,
                num_bands=cfg.resolved_num_bands,
                shingle_size=cfg.shingle_size,
                seed=cfg.seed,
                max_shingles=cfg.max_shingles,
            )

        if not self.use_pandas_udf:
            make_band_keys_py = udf(make_one, ArrayType(StringType()))
            return df.withColumn(output_col, make_band_keys_py(df[text_col]))

        @pandas_udf(ArrayType(StringType()))
        def make_band_keys(texts: pd.Series) -> pd.Series:
            return texts.fillna("").map(make_one)

        return df.withColumn(output_col, make_band_keys(df[text_col]))

    def _build_partition_bloom_from_text_rows(self, rows, text_col: str):
        cfg = self.config
        bloom = self._make_empty_bloom()
        for row in rows:
            try:
                text = row[text_col]
            except (KeyError, TypeError, ValueError):
                text = getattr(row, text_col)
            keys = band_keys_for_text(
                text=str(text) if text is not None else "",
                num_perm=cfg.num_perm,
                num_bands=cfg.resolved_num_bands,
                shingle_size=cfg.shingle_size,
                seed=cfg.seed,
                max_shingles=cfg.max_shingles,
            )
            for key in keys:
                bloom.add(key)
        yield bloom.to_bytes()

    @staticmethod
    def _or_merge_payloads(a: bytes, b: bytes) -> bytes:
        if not a:
            return b
        if not b:
            return a
        arr_a = np.frombuffer(a, dtype=np.uint8)
        arr_b = np.frombuffer(b, dtype=np.uint8)
        return np.bitwise_or(arr_a, arr_b).tobytes()

    def _build_bloom_from_df(self, df, text_col: str) -> MergeableBloomFilter:
        text_rdd = df.select(text_col).rdd
        payload_rdd = text_rdd.mapPartitions(lambda rows: self._build_partition_bloom_from_text_rows(rows, text_col))
        return self._merge_bloom_payload_rdd(payload_rdd, df.sparkSession)

    def _merge_bloom_payload_rdd(self, payload_rdd, spark_session) -> MergeableBloomFilter:
        is_local = str(spark_session.sparkContext.master).startswith("local")

        if is_local:
            payloads = payload_rdd.collect()
            if not payloads:
                return self._make_empty_bloom()
            merged_payload = payloads[0]
            for payload in payloads[1:]:
                merged_payload = self._or_merge_payloads(merged_payload, payload)
            template = self._make_empty_bloom()
            return MergeableBloomFilter.from_bytes(
                payload=merged_payload,
                expected_items=template.expected_items,
                fp_prob=template.fp_prob,
                num_bits=template.num_bits,
                num_hashes=template.num_hashes,
            )

        # Distributed tree-reduce: executors OR-merge bytes in pairs across log(N)
        # stages, so the driver never has to collect all partition Blooms at once.
        # If the RDD is empty (no rows survived filtering), fall back to a fresh
        # empty Bloom instead of crashing.
        try:
            merged_payload = payload_rdd.treeReduce(self._or_merge_payloads, depth=2)
        except ValueError:
            return self._make_empty_bloom()

        template = self._make_empty_bloom()
        return MergeableBloomFilter.from_bytes(
            payload=merged_payload,
            expected_items=template.expected_items,
            fp_prob=template.fp_prob,
            num_bits=template.num_bits,
            num_hashes=template.num_hashes,
        )

    def _band_table(self, df, text_col: str, id_col: str | None):
        from pyspark.sql import functions as F

        internal_id = id_col or "_spark_lshbloom_id"
        working = df if id_col else df.withColumn(internal_id, F.monotonically_increasing_id())
        band_df = self._band_keys_column(working, text_col=text_col)
        return band_df.select(
            F.col(internal_id).cast("string").alias("doc_id"),
            F.posexplode("band_keys").alias("band_id", "band_key"),
        )

    def fit(self, df, text_col: str = "text", id_col: str | None = None) -> "SparkLSHBloom":
        if self.mode != "lshbloom":
            raise ValueError("fit() is only available in mode='lshbloom'. Use deduplicate() for mode='minhash_lsh'.")

        self.bloom = self._build_bloom_from_df(df, text_col=text_col)
        return self

    def transform(
        self,
        df,
        text_col: str = "text",
        id_col: str | None = None,
        output_col: str = "is_duplicate",
    ):
        if self.mode != "lshbloom":
            raise ValueError("transform() is only available in mode='lshbloom'.")
        if self.bloom is None:
            raise ValueError("Bloom filter is not fitted. Call fit() or load() first.")

        from pyspark.sql.functions import pandas_udf, udf
        from pyspark.sql.types import BooleanType

        spark = df.sparkSession
        meta = self.bloom.metadata()
        bloom_kwargs = {k: meta[k] for k in ("expected_items", "fp_prob", "num_bits", "num_hashes")}
        payload = self.bloom.to_bytes()
        bc_meta = spark.sparkContext.broadcast(bloom_kwargs)
        bc_payload = spark.sparkContext.broadcast(payload)

        cfg = self.config

        def is_dup(text: str) -> bool:
            bloom = MergeableBloomFilter.from_bytes(payload=bc_payload.value, **bc_meta.value)
            keys = band_keys_for_text(
                text=str(text) if text is not None else "",
                num_perm=cfg.num_perm,
                num_bands=cfg.resolved_num_bands,
                shingle_size=cfg.shingle_size,
                seed=cfg.seed,
                max_shingles=cfg.max_shingles,
            )
            return any(key in bloom for key in keys)

        if not self.use_pandas_udf:
            query_bloom_py = udf(is_dup, BooleanType())
            return df.withColumn(output_col, query_bloom_py(df[text_col]))

        @pandas_udf(BooleanType())
        def query_bloom(texts: pd.Series) -> pd.Series:
            return texts.fillna("").map(is_dup)

        return df.withColumn(output_col, query_bloom(df[text_col]))

    def deduplicate(self, df, text_col: str = "text", id_col: str | None = None):
        if self.mode == "minhash_lsh":
            return self._deduplicate_banding(df, text_col=text_col, id_col=id_col)
        return self._deduplicate_bloom_chunked(df, text_col=text_col, id_col=id_col)

    def _deduplicate_banding(self, df, text_col: str, id_col: str | None):
        from pyspark.sql import functions as F

        internal_id = id_col or "_spark_lshbloom_id"
        working = df if id_col else df.withColumn(internal_id, F.monotonically_increasing_id())
        band_df = self._band_keys_column(working, text_col=text_col)

        exploded = band_df.select(internal_id, F.explode("band_keys").alias("band_key"))
        reps = exploded.groupBy("band_key").agg(F.min(internal_id).alias("_bucket_rep"))
        doc_reps = (
            exploded.join(reps, on="band_key", how="inner")
            .groupBy(internal_id)
            .agg(F.min("_bucket_rep").alias("_global_rep"))
        )
        unique_ids = doc_reps.where(F.col(internal_id) == F.col("_global_rep")).select(internal_id)
        result = working.join(unique_ids, on=internal_id, how="inner")
        return result.drop("_spark_lshbloom_id") if not id_col else result

    def _deduplicate_bloom_chunked(self, df, text_col: str, id_col: str | None):
        from functools import reduce

        from pyspark.sql import functions as F

        internal_id = id_col or "_spark_lshbloom_id"
        working = df if id_col else df.withColumn(internal_id, F.monotonically_increasing_id())
        chunks = self.config.deduplicate_chunks
        chunked = working.withColumn("_spark_lshbloom_chunk", F.pmod(F.abs(F.hash(F.col(internal_id))), F.lit(chunks)))

        self.bloom = self._make_empty_bloom()
        unique_chunks = []

        for chunk_id in range(chunks):
            chunk = chunked.where(F.col("_spark_lshbloom_chunk") == F.lit(chunk_id)).drop("_spark_lshbloom_chunk")
            queried = self.transform(chunk, text_col=text_col, id_col=internal_id)
            unique = queried.where(~F.col("is_duplicate")).drop("is_duplicate")
            unique_chunks.append(unique)
            self.bloom.merge(self._build_bloom_from_df(unique, text_col=text_col))

        if not unique_chunks:
            return working.limit(0)
        result = reduce(lambda left, right: left.unionByName(right), unique_chunks)
        return result.drop("_spark_lshbloom_id") if not id_col else result

    def save(self, path: str | Path) -> None:
        if self.bloom is None:
            raise ValueError("Cannot save before fit().")
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2), encoding="utf8")
        self.bloom.save(root / "bloom")

    @classmethod
    def load(cls, path: str | Path) -> "SparkLSHBloom":
        root = Path(path)
        config = LSHBloomConfig.from_dict(json.loads((root / "config.json").read_text(encoding="utf8")))
        model = cls(**config.to_dict())
        model.bloom = MergeableBloomFilter.load(root / "bloom")
        return model
