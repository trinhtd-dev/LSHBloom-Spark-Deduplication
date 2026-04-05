"""
Quick validator for a single JSON/JSONL input file against benchmark ingestion schema.

Usage:
  python synthetic_benchmark/test_input_json.py --file path/to/file.jsonl
  python synthetic_benchmark/test_input_json.py --file path/to/file.json --max-records 20
  python synthetic_benchmark/test_input_json.py --demo
  python synthetic_benchmark/test_input_json.py --inline-json "{\"doc_id\":\"x\",\"text\":\"hello\"}"
  python synthetic_benchmark/test_input_json.py --demo --show-structure
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

from dedup_benchmark_utils import normalize_record


SAMPLE_RECORD = {
    "doc_id": "2507.21563v3",
    "source_pdf": "arxiv_ocr_benchmark_workspace/pdfs/2507.21563v3.pdf",
    "parser_name": "pypdf",
    "text": "This is a short demo text.",
}


def iter_records(path: Path) -> Iterable[dict]:
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[INVALID_JSON] line={line_num} error={e}")
                    continue
                if not isinstance(data, dict):
                    print(f"[INVALID_RECORD] line={line_num} expected object")
                    continue
                yield data
        return

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Accept either a single object or a wrapped list under common keys.
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

    raise ValueError("Unsupported file extension. Use .json or .jsonl")


def summarize_record(rec: dict) -> str:
    path = rec.get("path", "")
    text = rec.get("text", "")
    doc_id = rec.get("doc_id", "")
    return f"doc_id={doc_id} path={str(path)[:80]} text_len={len(text)}"


def build_structure_example() -> dict:
    return {
        "accepted_minimal": {
            "path": "path/to/doc.txt",
            "text": "document content",
        },
        "accepted_arxiv_like": {
            "doc_id": "2507.21563v3",
            "source_pdf": "arxiv_ocr_benchmark_workspace/pdfs/2507.21563v3.pdf",
            "parser_name": "pypdf",
            "text": "full extracted text",
        },
        "optional_fields": ["metadata", "pages", "text_sha1", "char_count", "line_count"],
    }


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def _update_structure_stats(stats: Dict[str, Dict[str, Any]], value: Any, path: str = "") -> None:
    if path:
        stats[path]["count"] += 1
        stats[path]["types"].add(_type_name(value))

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            _update_structure_stats(stats, child, child_path)
        return

    if isinstance(value, list):
        array_path = f"{path}[]" if path else "[]"
        stats[array_path]["count"] += 1
        stats[array_path]["types"].add("array_items")
        for item in value:
            _update_structure_stats(stats, item, array_path)


def _print_scanned_structure(stats: Dict[str, Dict[str, Any]], scanned_records: int) -> None:
    print("\n=== Scanned File Structure ===")
    print(f"records_scanned={scanned_records}")

    top_level_keys = []
    for key in stats:
        if "." not in key and not key.endswith("[]"):
            top_level_keys.append(key)

    top_level_keys.sort()
    required_keys = [k for k in top_level_keys if stats[k]["count"] == scanned_records]
    optional_keys = [k for k in top_level_keys if stats[k]["count"] < scanned_records]

    print(f"top_level_required={required_keys}")
    print(f"top_level_optional={optional_keys}")
    print("fields:")

    for key in sorted(stats.keys()):
        types = ",".join(sorted(stats[key]["types"]))
        print(f"  - {key}: count={stats[key]['count']} types={types}")


def _blank_for_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _blank_for_value(v) for k, v in value.items()}
    if isinstance(value, list):
        # Keep arrays structurally valid and intentionally empty.
        return []
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Path to JSON/JSONL file")
    parser.add_argument("--inline-json", help="Inline JSON object string to validate")
    parser.add_argument("--demo", action="store_true", help="Validate a built-in sample JSON object")
    parser.add_argument("--show-structure", action="store_true", help="Print example input schema")
    parser.add_argument("--scan-structure", action="store_true", help="Scan actual keys/types from input records")
    parser.add_argument("--max-records", type=int, default=10, help="Maximum records to validate")
    args = parser.parse_args()

    if args.show_structure:
        print("=== Expected Input Structure ===")
        print(json.dumps(build_structure_example(), indent=2, ensure_ascii=True))
        print()

    selected_inputs = int(bool(args.file)) + int(bool(args.inline_json)) + int(bool(args.demo))
    if selected_inputs != 1:
        raise ValueError("Choose exactly one input mode: --file or --inline-json or --demo")

    records: Iterable[dict]
    source_name: str

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        records = iter_records(file_path)
        source_name = str(file_path)
    elif args.inline_json:
        try:
            obj = json.loads(args.inline_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid --inline-json: {e}") from e
        if not isinstance(obj, dict):
            raise ValueError("--inline-json must be a JSON object")
        records = [obj]
        source_name = "inline-json"
    else:
        records = [SAMPLE_RECORD]
        source_name = "demo-sample"

    total = 0
    valid = 0
    invalid = 0
    structure_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "types": set()})
    first_record_for_template: Optional[dict] = None

    for rec in records:
        if total >= args.max_records:
            break

        total += 1
        if args.scan_structure:
            _update_structure_stats(structure_stats, rec)
            if first_record_for_template is None:
                first_record_for_template = rec

        normalized: Optional[dict] = normalize_record(rec)
        if normalized is None:
            invalid += 1
            print(f"[INVALID_SCHEMA] #{total} missing usable path/text")
            continue

        valid += 1
        print(f"[OK] #{total} {summarize_record(normalized)}")

    print("\n=== Summary ===")
    print(f"source={source_name}")
    print(f"checked={total}")
    print(f"valid={valid}")
    print(f"invalid={invalid}")

    if args.scan_structure:
        _print_scanned_structure(structure_stats, total)
        if first_record_for_template is not None:
            print("\nempty_value_template:")
            print(json.dumps(_blank_for_value(first_record_for_template), indent=2, ensure_ascii=True))

    if total == 0:
        print("No records found. Check file format.")


if __name__ == "__main__":
    main()
