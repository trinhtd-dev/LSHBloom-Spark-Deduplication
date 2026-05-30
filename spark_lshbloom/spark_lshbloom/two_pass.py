from __future__ import annotations

from itertools import combinations

from pyspark.sql import DataFrame
from pyspark.sql.types import BooleanType, DoubleType, IntegerType, StringType, StructField, StructType

from .hashing import band_keys_for_text, char_ngrams


BAND_ROW_SCHEMA = StructType(
    [
        StructField("doc_id", StringType(), nullable=False),
        StructField("band_id", IntegerType(), nullable=False),
        StructField("band_key", StringType(), nullable=False),
    ]
)

PAIR_SCHEMA = StructType(
    [
        StructField("doc_id_a", StringType(), nullable=False),
        StructField("doc_id_b", StringType(), nullable=False),
    ]
)

SCORED_PAIR_SCHEMA = StructType(
    [
        StructField("doc_id_a", StringType(), nullable=False),
        StructField("doc_id_b", StringType(), nullable=False),
        StructField("jaccard", DoubleType(), nullable=False),
        StructField("is_duplicate", BooleanType(), nullable=False),
    ]
)


def emit_band_rows(
    df: DataFrame,
    id_col: str = "doc_id",
    text_col: str = "text",
    num_perm: int = 128,
    num_bands: int = 32,
    shingle_size: int = 5,
    seed: int = 42,
    max_shingles: int | None = 2048,
) -> DataFrame:
    """Emit one row per document-band key using RDD mapPartitions.

    This avoids materializing an intermediate Spark SQL array column of all
    band keys, which is expensive for PySpark on long text.
    """

    def rows_for_partition(rows):
        for row in rows:
            doc_id = str(row[id_col])
            text = row[text_col] if row[text_col] is not None else ""
            keys = band_keys_for_text(
                text=str(text),
                num_perm=num_perm,
                num_bands=num_bands,
                shingle_size=shingle_size,
                seed=seed,
                max_shingles=max_shingles,
            )
            for raw_key in keys:
                band_id_raw, band_hash = raw_key.split(":", 1)
                yield (doc_id, int(band_id_raw), band_hash)

    rdd = df.select(id_col, text_col).rdd.mapPartitions(rows_for_partition)
    return df.sparkSession.createDataFrame(rdd, schema=BAND_ROW_SCHEMA)


def detect_suspect_keys(
    band_rows_df: DataFrame,
    num_partitions: int = 64,
) -> DataFrame:
    """Detect repeated LSH bucket keys without collecting bucket doc IDs.

    Rows are partitioned by `(band_id, band_key)`, then each partition keeps a
    small local `seen` set and emits only keys observed at least twice. This is
    a two-pass recovery helper: it stores suspicious keys, not candidate IDs.
    """

    spark = band_rows_df.sparkSession

    keyed = band_rows_df.select("band_id", "band_key").rdd.map(
        lambda row: ((int(row.band_id), str(row.band_key)), 1)
    )
    partitioned = keyed.partitionBy(num_partitions)

    def detect_partition(rows):
        seen = set()
        emitted = set()
        for key, _ in rows:
            if key in seen:
                if key not in emitted:
                    emitted.add(key)
                    yield (key[0], key[1])
            else:
                seen.add(key)

    return spark.createDataFrame(partitioned.mapPartitions(detect_partition), schema="band_id int, band_key string")


def recover_candidate_pairs(
    band_rows_df: DataFrame,
    suspect_keys_df: DataFrame,
    max_bucket_size: int = 10_000,
) -> DataFrame:
    """Recover candidate document pairs from suspicious LSH buckets."""

    spark = band_rows_df.sparkSession
    candidate_rows = band_rows_df.join(suspect_keys_df, on=["band_id", "band_key"], how="inner")

    keyed = candidate_rows.select("band_id", "band_key", "doc_id").rdd.map(
        lambda row: ((int(row.band_id), str(row.band_key)), str(row.doc_id))
    )

    def create_combiner(doc_id: str):
        return {doc_id}

    def merge_value(values: set[str], doc_id: str):
        if len(values) <= max_bucket_size:
            values.add(doc_id)
        return values

    def merge_combiners(left: set[str], right: set[str]):
        if len(left) >= max_bucket_size:
            return left
        left.update(list(right)[: max_bucket_size - len(left)])
        return left

    buckets = keyed.combineByKey(create_combiner, merge_value, merge_combiners)

    def bucket_pairs(item):
        docs = sorted(item[1])
        if len(docs) < 2:
            return
        for left, right in combinations(docs, 2):
            yield (left, right)

    pairs = buckets.flatMap(bucket_pairs).distinct()
    return spark.createDataFrame(pairs, schema=PAIR_SCHEMA)


def recover_all_candidate_pairs(
    band_rows_df: DataFrame,
    max_bucket_size: int = 10_000,
) -> DataFrame:
    """Recover candidate pairs from all LSH buckets.

    This is the full distributed MinHashLSH-style baseline: every repeated bucket
    is materialized, unlike two-pass recovery which only groups suspect keys.
    """

    spark = band_rows_df.sparkSession
    keyed = band_rows_df.select("band_id", "band_key", "doc_id").rdd.map(
        lambda row: ((int(row.band_id), str(row.band_key)), str(row.doc_id))
    )

    def create_combiner(doc_id: str):
        return {doc_id}

    def merge_value(values: set[str], doc_id: str):
        if len(values) <= max_bucket_size:
            values.add(doc_id)
        return values

    def merge_combiners(left: set[str], right: set[str]):
        if len(left) >= max_bucket_size:
            return left
        left.update(list(right)[: max_bucket_size - len(left)])
        return left

    buckets = keyed.combineByKey(create_combiner, merge_value, merge_combiners)

    def bucket_pairs(item):
        docs = sorted(item[1])
        if len(docs) < 2:
            return
        for left, right in combinations(docs, 2):
            yield (left, right)

    pairs = buckets.flatMap(bucket_pairs).distinct()
    return spark.createDataFrame(pairs, schema=PAIR_SCHEMA)


def verify_candidate_pairs(
    pairs_df: DataFrame,
    docs_df: DataFrame,
    id_col: str = "doc_id",
    text_col: str = "text",
    shingle_size: int = 5,
    threshold: float = 0.85,
    max_shingles: int | None = 2048,
) -> DataFrame:
    """Score candidate pairs with exact character-shingle Jaccard similarity.

    LSHBloom and LSH banding only generate candidate pairs. This verification
    stage turns them into duplicate decisions by comparing the original texts on
    the much smaller candidate set.
    """

    spark = pairs_df.sparkSession
    docs = docs_df.selectExpr(f"cast({id_col} as string) as doc_id", f"{text_col} as text")

    left = docs.selectExpr("doc_id as doc_id_a", "text as text_a")
    right = docs.selectExpr("doc_id as doc_id_b", "text as text_b")
    joined = pairs_df.join(left, on="doc_id_a", how="inner").join(right, on="doc_id_b", how="inner")

    def score_partition(rows):
        for row in rows:
            shingles_a = set(char_ngrams(row.text_a or "", shingle_size, max_shingles=max_shingles))
            shingles_b = set(char_ngrams(row.text_b or "", shingle_size, max_shingles=max_shingles))
            if not shingles_a and not shingles_b:
                score = 1.0
            elif not shingles_a or not shingles_b:
                score = 0.0
            else:
                score = len(shingles_a & shingles_b) / float(len(shingles_a | shingles_b))
            yield (str(row.doc_id_a), str(row.doc_id_b), float(score), bool(score >= threshold))

    scored = joined.select("doc_id_a", "doc_id_b", "text_a", "text_b").rdd.mapPartitions(score_partition)
    return spark.createDataFrame(scored, schema=SCORED_PAIR_SCHEMA)
