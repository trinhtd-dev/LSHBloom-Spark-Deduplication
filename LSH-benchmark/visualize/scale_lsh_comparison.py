from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "benchmark_dfs"
STUB_DIR = Path(__file__).resolve().parent / "_stubs"
OUTPUT_DIR = Path(__file__).resolve().parent / "scaling"
DEFAULT_SOURCE_TAG = "test_p_0.2"
DEFAULT_SIZES = [1000, 2000, 4000, 6000, 8000, 10000]
DEFAULT_THRESHOLD = 0.5
DEFAULT_NUM_PERM = 256
DEFAULT_B_BITS = 8
DEFAULT_NUM_PROBES = 8
DEFAULT_WORKERS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure scaling of LSH variants on benchmark_dfs.")
    parser.add_argument("--source-tag", default=DEFAULT_SOURCE_TAG, help="Base benchmark tag to sample from.")
    parser.add_argument("--sizes", default="1000,2000,4000,6000,8000,10000", help="Comma-separated sample sizes.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Similarity threshold for all runs.")
    parser.add_argument("--num-perm", type=int, default=DEFAULT_NUM_PERM)
    parser.add_argument("--b-bits", type=int, default=DEFAULT_B_BITS)
    parser.add_argument("--num-probes", type=int, default=DEFAULT_NUM_PROBES)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel benchmark workers.")
    parser.add_argument(
        "--algorithms",
        default="lsh,bbit_lsh,lsh_multiprobe",
        help="Comma-separated algorithms to run.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory to write CSV/SVG outputs.",
    )
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_jsonl_docs(path: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text")
            if not isinstance(text, str) or text is None or text.strip() == "" or text == "None":
                continue
            docs.append(obj)
    return docs


def build_subset(source_tag: str, size: int) -> str:
    source_jsonl = DATA_DIR / f"{source_tag}.jsonl"
    source_csv = DATA_DIR / f"{source_tag}.csv"
    if not source_jsonl.exists() or not source_csv.exists():
        raise FileNotFoundError(f"Missing source benchmark files for {source_tag}")

    subset_tag = f"scale_{source_tag}_n{size}"
    subset_jsonl = DATA_DIR / f"{subset_tag}.jsonl"
    subset_csv = DATA_DIR / f"{subset_tag}.csv"

    docs = read_jsonl_docs(source_jsonl)
    if len(docs) < size:
        raise ValueError(f"Requested size {size} exceeds available docs {len(docs)} in {source_tag}")
    selected = docs[:size]
    selected_ids = {str(doc["id"]) for doc in selected}

    with subset_jsonl.open("w", encoding="utf-8") as f:
        for doc in selected:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    with source_csv.open(newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="|")
        rows = [row for row in reader if str(row["id"]) in selected_ids]

    with subset_csv.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames or [], delimiter="|")
        writer.writeheader()
        writer.writerows(rows)

    return subset_tag


def available_doc_count(source_tag: str) -> int:
    source_jsonl = DATA_DIR / f"{source_tag}.jsonl"
    if not source_jsonl.exists():
        raise FileNotFoundError(f"Missing source benchmark file for {source_tag}")
    return len(read_jsonl_docs(source_jsonl))


def algorithm_command(algo: str, subset_tag: str, threshold: float, num_perm: int, b_bits: int, num_probes: int) -> tuple[str, list[str]]:
    if algo == "lsh":
        return (
            "dedup/lsh/lsh.py",
            [
                "--input",
                subset_tag,
                "--sim-threshold",
                f"{threshold:.1f}",
                "--num-perm",
                str(num_perm),
                "--ngram",
                "1",
            ],
        )
    if algo == "bbit_lsh":
        return (
            "dedup/minhash_bbit_lsh/bbit_lsh.py",
            [
                "--input",
                subset_tag,
                "--threshold",
                f"{threshold:.1f}",
                "--num-perm",
                str(num_perm),
                "--b-bits",
                str(b_bits),
                "--shingle-size",
                "1",
            ],
        )
    if algo == "lsh_multiprobe":
        return (
            "dedup/lsh/lsh_multiprobe.py",
            [
                "--input",
                subset_tag,
                "--sim-threshold",
                f"{threshold:.1f}",
                "--num-perm",
                str(num_perm),
                "--ngram",
                "1",
                "--num-probes",
                str(num_probes),
            ],
        )
    raise ValueError(f"Unknown algorithm: {algo}")


def run_case(algo: str, subset_tag: str, size: int, threshold: float, num_perm: int, b_bits: int, num_probes: int) -> dict[str, Any]:
    script_path, script_args = algorithm_command(algo, subset_tag, threshold, num_perm, b_bits, num_probes)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(STUB_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    wrapper_code = f"""
import json
import runpy
import sys
import time
import tracemalloc

sys.argv = {[script_path, *script_args]!r}
tracemalloc.start()
t0 = time.perf_counter()
runpy.run_path({repr(str((BASE_DIR / script_path).resolve()))}, run_name="__main__")
runtime = time.perf_counter() - t0
peak = tracemalloc.get_traced_memory()[1] / 1024 / 1024
print("__BENCHMARK_JSON__" + json.dumps({{"runtime_sec": runtime, "peak_tracemalloc_mb": peak}}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", wrapper_code],
        cwd=BASE_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"{algo} failed on {subset_tag}.\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    marker = "__BENCHMARK_JSON__"
    payload_line = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.startswith(marker)),
        None,
    )
    if not payload_line:
        raise RuntimeError(
            f"{algo} on {subset_tag} did not emit benchmark metadata.\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    payload = json.loads(payload_line[len(marker) :])

    return {
        "algorithm": algo,
        "dataset": subset_tag,
        "size": size,
        "threshold": threshold,
        "runtime_sec": round(float(payload["runtime_sec"]), 6),
        "peak_tracemalloc_mb": round(float(payload["peak_tracemalloc_mb"]), 6),
    }


def make_svg_line_chart(
    title: str,
    x_values: list[int],
    series: dict[str, list[float]],
    y_label: str,
    output_path: Path,
) -> None:
    width = 1100
    height = 680
    margin_left = 95
    margin_right = 30
    margin_top = 70
    margin_bottom = 85
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    values = [value for points in series.values() for value in points]
    if not values:
        raise ValueError("No data to plot")
    y_min = min(values)
    y_max = max(values)
    if math.isclose(y_min, y_max):
        y_min -= 0.05
        y_max += 0.05
    pad = max((y_max - y_min) * 0.08, 0.05)
    y_min -= pad
    y_max += pad

    def x_scale(x: float) -> float:
        if len(x_values) == 1:
            return margin_left + plot_w / 2
        return margin_left + ((x - x_values[0]) / (x_values[-1] - x_values[0])) * plot_w

    def y_scale(y: float) -> float:
        return margin_top + (y_max - y) / (y_max - y_min) * plot_h

    def fmt(num: float) -> str:
        return f"{num:.2f}".rstrip("0").rstrip(".")

    palette = {
        "lsh": "#1f77b4",
        "bbit_lsh": "#d62728",
        "lsh_multiprobe": "#2ca02c",
    }
    markers = {
        "lsh": "o",
        "bbit_lsh": "s",
        "lsh_multiprobe": "^",
    }

    y_grid = []
    for i in range(6):
        y = margin_top + (plot_h * i / 5)
        value = y_max - ((y - margin_top) / plot_h) * (y_max - y_min)
        y_grid.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e6e6e6" stroke-width="1" />'
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#666">{fmt(value)}</text>'
        )

    x_ticks = []
    for x in x_values:
        sx = x_scale(x)
        x_ticks.append(
            f'<line x1="{sx:.2f}" y1="{height - margin_bottom}" x2="{sx:.2f}" y2="{height - margin_bottom + 6}" stroke="#666" stroke-width="1" />'
            f'<text x="{sx:.2f}" y="{height - margin_bottom + 24}" text-anchor="middle" font-size="12" fill="#666">{x}</text>'
        )

    plot_lines = []
    legend_items = []
    legend_x = width - margin_right - 260
    legend_y = margin_top - 38
    for idx, (algo, values_for_algo) in enumerate(series.items()):
        points = [(x_scale(x), y_scale(y)) for x, y in zip(x_values, values_for_algo)]
        path = " ".join([f"M {points[0][0]:.2f} {points[0][1]:.2f}"] + [f"L {x:.2f} {y:.2f}" for x, y in points[1:]])
        circles = "".join(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{palette[algo]}" stroke="white" stroke-width="1.5" />'
            for x, y in points
        )
        plot_lines.append(
            f'<path d="{path}" fill="none" stroke="{palette[algo]}" stroke-width="3" />{circles}'
        )
        y = legend_y + idx * 24
        legend_items.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 20}" y2="{y}" stroke="{palette[algo]}" stroke-width="3" />'
            f'<circle cx="{legend_x + 10}" cy="{y}" r="4" fill="{palette[algo]}" stroke="white" stroke-width="1" />'
            f'<text x="{legend_x + 30}" y="{y + 4}" font-size="13" fill="#333">{algo}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{width / 2:.2f}" y="34" text-anchor="middle" font-size="24" font-weight="600" fill="#222">{title}</text>
  <text x="{width / 2:.2f}" y="{height - 20}" text-anchor="middle" font-size="14" fill="#444">data size</text>
  <text x="24" y="{height / 2:.2f}" text-anchor="middle" font-size="14" fill="#444" transform="rotate(-90 24 {height / 2:.2f})">{y_label}</text>
  <rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#d9d9d9" />
  {''.join(y_grid)}
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#666" stroke-width="1.2" />
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#666" stroke-width="1.2" />
  {''.join(x_ticks)}
  {''.join(plot_lines)}
  {''.join(legend_items)}
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    args = parse_args()
    requested_sizes = parse_int_list(args.sizes)
    algorithms = parse_str_list(args.algorithms)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    available = available_doc_count(args.source_tag)
    if any(size > available for size in requested_sizes):
        sizes = sorted(
            {
                max(1, int(available * frac))
                for frac in (0.2, 0.4, 0.6, 0.8, 1.0)
            }
        )
        print(f"[warn] requested sizes exceed available docs ({available}); using auto sizes: {sizes}")
    else:
        sizes = sorted(set(requested_sizes))

    for size in sizes:
        build_subset(args.source_tag, size)

    cases = []
    for size in sizes:
        subset_tag = f"scale_{args.source_tag}_n{size}"
        for algo in algorithms:
            cases.append((algo, subset_tag, size))

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(
                run_case,
                algo,
                subset_tag,
                size,
                args.threshold,
                args.num_perm,
                args.b_bits,
                args.num_probes,
            ): (algo, size)
            for algo, subset_tag, size in cases
        }
        for future in as_completed(future_map):
            results.append(future.result())

    results.sort(key=lambda row: (row["size"], row["algorithm"]))

    csv_path = output_dir / "scale_benchmark_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["algorithm", "dataset", "size", "threshold", "runtime_sec", "peak_tracemalloc_mb"],
        )
        writer.writeheader()
        writer.writerows(results)

    by_algo_runtime: dict[str, list[float]] = {algo: [] for algo in algorithms}
    by_algo_mem: dict[str, list[float]] = {algo: [] for algo in algorithms}
    for size in sizes:
        for algo in algorithms:
            row = next(item for item in results if item["size"] == size and item["algorithm"] == algo)
            by_algo_runtime[algo].append(row["runtime_sec"])
            by_algo_mem[algo].append(row["peak_tracemalloc_mb"])

    make_svg_line_chart(
        title=f"Runtime scaling at threshold={args.threshold:.1f}",
        x_values=sizes,
        series=by_algo_runtime,
        y_label="runtime (sec)",
        output_path=output_dir / "scale_runtime_by_size.svg",
    )
    make_svg_line_chart(
        title=f"Peak RAM scaling at threshold={args.threshold:.1f}",
        x_values=sizes,
        series=by_algo_mem,
        y_label="peak traced memory (MB)",
        output_path=output_dir / "scale_ram_by_size.svg",
    )

    print(f"Saved: {csv_path}")
    print(f"Saved: {output_dir / 'scale_runtime_by_size.svg'}")
    print(f"Saved: {output_dir / 'scale_ram_by_size.svg'}")


if __name__ == "__main__":
    main()
