from pyspark.sql import SparkSession

from spark_lshbloom import SparkLSHBloom


def main():
    spark = (
        SparkSession.builder.master("local[4]")
        .appName("spark-lshbloom-demo")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    df = spark.createDataFrame(
        [
            (1, "internet scale text deduplication with bloom filters"),
            (2, "internet-scale text deduplication with bloom filter"),
            (3, "a completely different scientific document"),
            (4, "another unrelated paper about distributed systems"),
        ],
        ["id", "text"],
    )

    lsh = SparkLSHBloom(threshold=0.85, num_perm=64, num_bands=16, expected_items=1000, mode="lshbloom")
    unique_df = lsh.deduplicate(df, text_col="text", id_col="id")
    unique_df.show(truncate=False)

    lsh.save("runs/demo_lshbloom")
    loaded = SparkLSHBloom.load("runs/demo_lshbloom")
    loaded.transform(df, text_col="text", id_col="id").show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
