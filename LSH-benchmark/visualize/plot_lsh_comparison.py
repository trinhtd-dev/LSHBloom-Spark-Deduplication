from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
COMPARE_JSON = BASE_DIR / "bbit_vs_lsh_compare.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_NUM_PERM = 128
DEFAULT_NUM_PROBES = 8
DEFAULT_METRIC = "f1"

ALGO_ORDER = ["lsh", "bbit_lsh", "lsh_multiprobe"]
ALGO_LABELS = {
    "lsh": "lsh",
    "bbit_lsh": "bbit_lsh",
    "lsh_multiprobe": "lsh_multiprobe",
}
ALGO_COLORS = {
    "lsh": "#1f77b4",
    "bbit_lsh": "#d62728",
    "lsh_multiprobe": "#2ca02c",
}
ALGO_MARKERS = {
    "lsh": "o",
    "bbit_lsh": "s",
    "lsh_multiprobe": "^",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LSH comparison charts.")
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help="Metric to plot. Default: f1",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write the PNG files.",
    )
    parser.add_argument(
        "--compare-json",
        default=str(COMPARE_JSON),
        help="Path to bbit_vs_lsh_compare.json.",
    )
    parser.add_argument(
        "--num-perm",
        type=int,
        default=DEFAULT_NUM_PERM,
        help="Num perm used for Multi-Probe score files.",
    )
    parser.add_argument(
        "--num-probes",
        type=int,
        default=DEFAULT_NUM_PROBES,
        help="Num probes used for Multi-Probe score files.",
    )
    return parser.parse_args()


def load_compare_rows(compare_json: Path) -> list[dict[str, Any]]:
    with compare_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload["cases"])


def load_multiprobe_rows(base_dir: Path, num_perm: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_dir in sorted(base_dir.glob("test_p_*")):
        if not dataset_dir.is_dir():
            continue
        try:
            p_value = float(dataset_dir.name.replace("test_p_", ""))
        except ValueError:
            continue

        result_dir = dataset_dir / "lsh_multiprobe_results"
        if not result_dir.exists():
            continue

        for score_file in sorted(result_dir.glob(f"lsh_multiprobe_*_{num_perm}_score.csv")):
            threshold_text = score_file.name.split("_")[2]
            try:
                threshold = float(threshold_text)
            except ValueError:
                continue

            with score_file.open(newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f))

            rows.append(
                {
                    "dataset": dataset_dir.name,
                    "duplicate_prevalence": p_value,
                    "threshold": threshold,
                    "metric": {
                        "precision": float(row["precision"]),
                        "recall": float(row["recall"]),
                        "f1": float(row["f1"]),
                        "acc": float(row["acc"]),
                        "bal_acc": float(row["bal_acc"]),
                    },
                }
            )
    return rows


def aggregate_compare_rows(rows: list[dict[str, Any]], algo: str, metric: str) -> dict[float, dict[float, float]]:
    buckets: dict[float, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        p = float(row["duplicate_prevalence"])
        threshold = float(row["threshold"])
        value = float(row[algo][metric])
        buckets[p][threshold].append(value)

    out: dict[float, dict[float, float]] = {}
    for p, by_threshold in buckets.items():
        out[p] = {thr: sum(values) / len(values) for thr, values in by_threshold.items()}
    return out


def aggregate_multiprobe_rows(rows: list[dict[str, Any]], metric: str) -> dict[float, dict[float, float]]:
    buckets: dict[float, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        p = float(row["duplicate_prevalence"])
        threshold = float(row["threshold"])
        value = float(row["metric"][metric])
        buckets[p][threshold].append(value)

    out: dict[float, dict[float, float]] = {}
    for p, by_threshold in buckets.items():
        out[p] = {thr: sum(values) / len(values) for thr, values in by_threshold.items()}
    return out


def aggregate_by_threshold(per_p: dict[float, dict[float, float]]) -> dict[float, float]:
    by_threshold: dict[float, list[float]] = defaultdict(list)
    for p_map in per_p.values():
        for threshold, value in p_map.items():
            by_threshold[threshold].append(value)
    return {thr: sum(values) / len(values) for thr, values in by_threshold.items()}


def aggregate_by_prevalence(per_p: dict[float, dict[float, float]]) -> dict[float, float]:
    return {p: sum(threshold_map.values()) / len(threshold_map) for p, threshold_map in per_p.items()}


def plot_line_chart(
    x_values: list[float],
    series: dict[str, list[float]],
    title: str,
    x_label: str,
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

    all_values = [value for values in series.values() for value in values]
    if not all_values:
        raise ValueError("No values to plot")
    y_min = min(all_values)
    y_max = max(all_values)
    if math.isclose(y_min, y_max):
        y_min -= 0.05
        y_max += 0.05
    pad = (y_max - y_min) * 0.08
    y_min -= pad
    y_max += pad

    def scale_x(x: float) -> float:
        if len(x_values) == 1:
            return margin_left + plot_w / 2
        x0, x1 = x_values[0], x_values[-1]
        return margin_left + ((x - x0) / (x1 - x0)) * plot_w

    def scale_y(y: float) -> float:
        return margin_top + (y_max - y) / (y_max - y_min) * plot_h

    def fmt(num: float) -> str:
        return f"{num:.4f}".rstrip("0").rstrip(".")

    grid_lines = []
    for i in range(6):
        y = margin_top + (plot_h * i / 5)
        value = y_max - ((y - margin_top) / plot_h) * (y_max - y_min)
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e6e6e6" stroke-width="1" />'
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#666">{fmt(value)}</text>'
        )

    x_ticks = []
    for x in x_values:
        sx = scale_x(x)
        x_ticks.append(
            f'<line x1="{sx:.2f}" y1="{height - margin_bottom}" x2="{sx:.2f}" y2="{height - margin_bottom + 6}" stroke="#666" stroke-width="1" />'
            f'<text x="{sx:.2f}" y="{height - margin_bottom + 24}" text-anchor="middle" font-size="12" fill="#666">{x:.1f}</text>'
        )

    plot_lines = []
    for algo in ALGO_ORDER:
        if algo not in series:
            continue
        points = []
        for x, y in zip(x_values, series[algo]):
            points.append((scale_x(x), scale_y(y)))
        path = " ".join([f"M {points[0][0]:.2f} {points[0][1]:.2f}"] + [f"L {x:.2f} {y:.2f}" for x, y in points[1:]])
        circles = "".join(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{ALGO_COLORS[algo]}" stroke="white" stroke-width="1.5" />'
            for x, y in points
        )
        plot_lines.append(
            f'<path d="{path}" fill="none" stroke="{ALGO_COLORS[algo]}" stroke-width="3" />{circles}'
        )

    legend_items = []
    legend_x = width - margin_right - 260
    legend_y = margin_top - 38
    for i, algo in enumerate(ALGO_ORDER):
        if algo not in series:
            continue
        y = legend_y + i * 24
        legend_items.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 20}" y2="{y}" stroke="{ALGO_COLORS[algo]}" stroke-width="3" />'
            f'<circle cx="{legend_x + 10}" cy="{y}" r="4" fill="{ALGO_COLORS[algo]}" stroke="white" stroke-width="1" />'
            f'<text x="{legend_x + 30}" y="{y + 4}" font-size="13" fill="#333">{ALGO_LABELS[algo]}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{width / 2:.2f}" y="34" text-anchor="middle" font-size="24" font-weight="600" fill="#222">{title}</text>
  <text x="{width / 2:.2f}" y="{height - 20}" text-anchor="middle" font-size="14" fill="#444">{x_label}</text>
  <text x="24" y="{height / 2:.2f}" text-anchor="middle" font-size="14" fill="#444" transform="rotate(-90 24 {height / 2:.2f})">{y_label}</text>
  <rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#d9d9d9" />
  {''.join(grid_lines)}
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#666" stroke-width="1.2" />
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#666" stroke-width="1.2" />
  {''.join(x_ticks)}
  {''.join(plot_lines)}
  {''.join(legend_items)}
</svg>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def build_chart_data(
    compare_rows: list[dict[str, Any]],
    multiprobe_rows: list[dict[str, Any]],
    metric: str,
) -> tuple[
    list[float],
    dict[str, list[float]],
    list[float],
    dict[str, list[float]],
    ]:
    compare_per_p = {
        algo: aggregate_compare_rows(compare_rows, algo, metric)
        for algo in ["lsh", "bbit_lsh"]
    }
    multiprobe_per_p = aggregate_multiprobe_rows(multiprobe_rows, metric)

    compare_by_p = {algo: aggregate_by_prevalence(per_p) for algo, per_p in compare_per_p.items()}
    multiprobe_by_p = aggregate_by_prevalence(multiprobe_per_p)

    p_values = sorted(set(compare_by_p["lsh"].keys()) | set(compare_by_p["bbit_lsh"].keys()) | set(multiprobe_by_p.keys()))
    p_series = {
        "lsh": [compare_by_p["lsh"][p] for p in p_values],
        "bbit_lsh": [compare_by_p["bbit_lsh"][p] for p in p_values],
        "lsh_multiprobe": [multiprobe_by_p[p] for p in p_values],
    }

    compare_by_threshold = {
        algo: aggregate_by_threshold(per_p)
        for algo, per_p in compare_per_p.items()
    }
    multiprobe_by_threshold = aggregate_by_threshold(multiprobe_per_p)

    threshold_values = sorted(
        set(compare_by_threshold["lsh"].keys())
        | set(compare_by_threshold["bbit_lsh"].keys())
        | set(multiprobe_by_threshold.keys())
    )
    threshold_series = {
        "lsh": [compare_by_threshold["lsh"][t] for t in threshold_values],
        "bbit_lsh": [compare_by_threshold["bbit_lsh"][t] for t in threshold_values],
        "lsh_multiprobe": [multiprobe_by_threshold[t] for t in threshold_values],
    }

    return p_values, p_series, threshold_values, threshold_series


def main() -> None:
    args = parse_args()
    metric = args.metric.strip().lower()
    if metric not in {"precision", "recall", "f1", "acc", "bal_acc"}:
        raise ValueError("metric must be one of: precision, recall, f1, acc, bal_acc")

    compare_json = Path(args.compare_json)
    output_dir = Path(args.output_dir)

    compare_rows = load_compare_rows(compare_json)
    multiprobe_rows = load_multiprobe_rows(BASE_DIR, args.num_perm)
    if not multiprobe_rows:
        raise FileNotFoundError("No Multi-Probe score files found under LSH-benchmark/test_p_*/lsh_multiprobe_results")

    p_values, p_series, threshold_values, threshold_series = build_chart_data(
        compare_rows=compare_rows,
        multiprobe_rows=multiprobe_rows,
        metric=metric,
    )

    plot_line_chart(
        x_values=p_values,
        series=p_series,
        title=f"{metric.upper()} by duplicate prevalence (p)",
        x_label="duplicate prevalence p",
        y_label=metric,
        output_path=output_dir / f"comparison_by_p_{metric}.svg",
    )
    plot_line_chart(
        x_values=threshold_values,
        series=threshold_series,
        title=f"{metric.upper()} by threshold",
        x_label="threshold",
        y_label=metric,
        output_path=output_dir / f"comparison_by_threshold_{metric}.svg",
    )

    print(f"Saved: {output_dir / f'comparison_by_p_{metric}.svg'}")
    print(f"Saved: {output_dir / f'comparison_by_threshold_{metric}.svg'}")


if __name__ == "__main__":
    main()
