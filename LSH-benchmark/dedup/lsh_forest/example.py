import tempfile

from lsh_forest import MinHashLSHForestDeduper


def main() -> None:
    docs = [
        ("1", "the quick brown fox jumps over the lazy dog"),
        ("2", "the quick brown fox jumps over a lazy dog"),
        ("3", "graph neural networks for molecule property prediction"),
        ("4", "the quick brown fox jumps over the lazy dog again"),
    ]

    with tempfile.TemporaryDirectory() as minhash_root:
        deduper = MinHashLSHForestDeduper(
            sim_threshold=0.6,
            num_perm=64,
            shingle_size=2,
            minhash_root=minhash_root,
            num_trees=8,
            top_k=3,
        )

        for doc_id, text in docs:
            print(doc_id, deduper.deduplicate(text, int(doc_id)))

        print(deduper.summary())


if __name__ == "__main__":
    main()
