"""
Extract only id and text fields from JSON/JSONL records for dedup pipelines.

Examples:
  python synthetic_benchmark/extract_id_text_jsonl.py \
    --input archive/benchmark_cloud_package/data/pypdf.jsonl \
    --output synthetic_benchmark/pypdf_id_text.jsonl

  python synthetic_benchmark/extract_id_text_jsonl.py \
    --input synthetic_benchmark/test.jsonl \
    --output synthetic_benchmark/test_id_text.jsonl \
    --max-records 10
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from dedup_benchmark_utils import normalize_record


def iter_records(path: Path) -> Iterable[dict]:
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[INVALID_JSON] line={line_num} error={e}")
                    continue
                if not isinstance(rec, dict):
                    print(f"[INVALID_RECORD] line={line_num} expected object")
                    continue
                yield rec
        return

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            if "records" in data and isinstance(data["records"], list):
                for rec in data["records"]:
                    if isinstance(rec, dict):
                        yield rec
                return
            yield data
            return

        if isinstance(data, list):
            for idx, rec in enumerate(data, start=1):
                if not isinstance(rec, dict):
                    print(f"[INVALID_RECORD] index={idx} expected object")
                    continue
                yield rec
            return

        raise ValueError("JSON root must be object or array")

    raise ValueError("Unsupported extension. Use .json or .jsonl")


def choose_id(rec: dict, fallback_index: int) -> str:
    if isinstance(rec.get("id"), (str, int)):
        return str(rec["id"])
    if isinstance(rec.get("doc_id"), str) and rec["doc_id"].strip():
        return rec["doc_id"]
    return str(fallback_index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON/JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL path (id/text only)")
    parser.add_argument("--max-records", type=int, default=0, help="Limit processed records (0 = all)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists() and in_path.suffix.lower() == ".json":
        alt = in_path.with_suffix(".jsonl")
        if alt.exists():
            print(f"[INFO] input not found: {in_path}; using {alt}")
            in_path = alt

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = set()
    total = 0
    written = 0
    skipped = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for rec in iter_records(in_path):
            if args.max_records > 0 and total >= args.max_records:
                break

            total += 1
            normalized: Optional[dict] = normalize_record(rec)
            if normalized is None:
                skipped += 1
                continue

            text = normalized.get("text")
            if not isinstance(text, str) or text.strip() == "":
                skipped += 1
                continue

            rec_id = choose_id(normalized, total)
            if rec_id in seen_ids:
                rec_id = f"{rec_id}__dup{total}"
            seen_ids.add(rec_id)

            out_obj = {
                "id": rec_id, 
                "text": text,
                "meta": {
                    "doc_id": normalized.get("doc_id", ""),
                    "parser_name": normalized.get("parser_name", ""),
                    "source_type": normalized.get("source_type", ""),
                    "title": normalized.get("title", ""),
                    "primary_category": normalized.get("primary_category", ""),
                    "published": normalized.get("published", ""),
                    "metadata": normalized.get("metadata", {})
                }
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            written += 1

    print("=== Extraction Summary ===")
    print(f"input={in_path}")
    print(f"output={out_path}")
    print(f"processed={total}")
    print(f"written={written}")
    print(f"skipped={skipped}")


if __name__ == "__main__":
    main()
