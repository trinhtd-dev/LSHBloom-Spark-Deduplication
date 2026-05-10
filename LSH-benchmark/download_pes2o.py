import argparse
import gc
import hashlib
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


def exact_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def batched_iter(iterable, batch_size: int):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_chunk(rows, part_idx: int, out_dir: Path, schema: pa.Schema):
    table = pa.Table.from_pylist(rows, schema=schema)
    out_path = out_dir / f"part-{part_idx:05d}.parquet"
    pq.write_table(table, out_path, compression="zstd")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a peS2o subset and save as parquet shards.")
    parser.add_argument("--dataset", default="allenai/peS2o", help="Hugging Face dataset name.")
    parser.add_argument("--config", default="v2", help="Dataset config.")
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument("--target-docs", type=int, default=50_000, help="Number of docs to keep.")
    parser.add_argument("--source-substr", default="", help="Filter: keep rows where source contains substring.")
    parser.add_argument("--min-words", type=int, default=0, help="Filter: minimum words in text.")
    parser.add_argument("--read-batch-size", type=int, default=256, help="Streaming read batch size.")
    parser.add_argument("--rows-per-file", type=int, default=5_000, help="Rows per parquet shard.")
    parser.add_argument(
        "--out-dir",
        default=str(Path("data") / "pes2o_50k"),
        help="Output directory for parquet shards.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = pa.schema(
        [
            ("doc_id", pa.int64()),
            ("source", pa.string()),
            ("text", pa.string()),
            ("n_words", pa.int32()),
            ("n_chars", pa.int32()),
            ("exact_hash", pa.string()),
        ]
    )

    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=True,
        trust_remote_code=True,
    )

    buffer = []
    part_idx = 0
    kept = 0
    seen = 0
    t0 = time.perf_counter()

    for ex_batch in batched_iter(dataset, args.read_batch_size):
        seen += len(ex_batch)
        for ex in ex_batch:
            src = str(ex.get("source", "")).lower()
            if args.source_substr and args.source_substr.lower() not in src:
                continue
            text = ex.get("text")
            if not text or not isinstance(text, str):
                continue
            n_words = len(text.split())
            if n_words < args.min_words:
                continue

            buffer.append(
                {
                    "doc_id": kept,
                    "source": src,
                    "text": text,
                    "n_words": n_words,
                    "n_chars": len(text),
                    "exact_hash": exact_md5(text),
                }
            )
            kept += 1
            if kept >= args.target_docs:
                break

        while len(buffer) >= args.rows_per_file:
            write_chunk(buffer[: args.rows_per_file], part_idx, out_dir, schema)
            buffer = buffer[args.rows_per_file :]
            part_idx += 1
            gc.collect()

        if kept and kept % 10_000 < args.read_batch_size:
            elapsed = time.perf_counter() - t0
            print(f"[kept={kept:,}] [seen={seen:,}] [elapsed={elapsed/60:.1f} min]")

        if kept >= args.target_docs:
            break

    if buffer:
        write_chunk(buffer, part_idx, out_dir, schema)

    elapsed = time.perf_counter() - t0
    print(f"Done. kept={kept:,}, seen={seen:,}, elapsed={elapsed/60:.2f} min")
    print("Output dir:", out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
