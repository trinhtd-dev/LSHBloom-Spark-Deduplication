from bbit_lsh import BBitMinHashLSH


def main() -> None:
    docs = [
        ("d1", "the quick brown fox jumps over the lazy dog"),
        ("d2", "the quick brown fox jumps over a lazy dog"),
        ("d3", "graph neural networks for molecule property prediction"),
        ("d4", "the quick brown fox jumps over the lazy dog again"),
    ]

    index = BBitMinHashLSH(
        threshold=0.7,
        num_perm=64,
        b_bits=8,
        shingle_size=2,
        num_bands=16,
        rows_per_band=4,
    )

    for doc_id, text in docs:
        is_dup, matches = index.query_and_insert(doc_id, text)
        print(f"{doc_id}: duplicate={is_dup} matches={matches}")

    print(index.summary())


if __name__ == "__main__":
    main()
