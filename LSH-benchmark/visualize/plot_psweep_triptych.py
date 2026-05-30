from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = BASE_DIR / "psweep_summary.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "psweep_triptych.png"
METRICS = ["precision", "recall", "f1"]

COLOR_CYCLE = [
    "#ff6b81",
    "#b8860b",
    "#2ca02c",
    "#17becf",
    "#1f77b4",
    "#e377c2",
    "#9467bd",
]
MARKERS = ["o", "s", "D", "^", "v", "P", "X"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Precision/Recall/F1 vs duplicate proportion at one threshold."
    )
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument(
        "--algos",
        default="",
        help="Comma-separated algorithm list. Default = all in summary.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def unique_sorted(values: Iterable[float]) -> list[float]:
    return sorted({float(v) for v in values})


def build_series(
    rows: list[dict[str, str]],
    algos: list[str],
    p_values: list[float],
    metric: str,
) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}
    for algo in algos:
        values: list[float] = []
        for p in p_values:
            match = next(
                (
                    r
                    for r in rows
                    if r.get("algorithm") == algo
                    and float(r.get("p", 0)) == p
                ),
                None,
            )
            if match is None:
                values.append(float("nan"))
            else:
                values.append(float(match.get(metric, 0) or 0))
        series[algo] = values
    return series


def write_metric_csv(
    output_path: Path,
    p_values: list[float],
    algos: list[str],
    rows: list[dict[str, str]],
    metric: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["p", "algorithm", metric])
        for p in p_values:
            for algo in algos:
                match = next(
                    (
                        r
                        for r in rows
                        if r.get("algorithm") == algo
                        and float(r.get("p", 0)) == p
                    ),
                    None,
                )
                value = float(match.get(metric, 0) or 0) if match else float("nan")
                writer.writerow([p, algo, value])


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    threshold = float(f"{args.threshold:.1f}")

    rows = load_rows(summary_path)
    rows = [r for r in rows if float(r.get("threshold", 0)) == threshold]
    if not rows:
        raise SystemExit(f"No rows found at threshold={threshold:.1f}")

    p_values = unique_sorted(r.get("p", 0) for r in rows)

    if args.algos.strip():
        algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    else:
        algos = sorted({r.get("algorithm", "") for r in rows})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        series = build_series(rows, algos, p_values, metric)
        for i, algo in enumerate(algos):
            color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
            marker = MARKERS[i % len(MARKERS)]
            ys = np.array(series[algo], dtype=float)
            ax.plot(
                p_values,
                ys,
                marker=marker,
                linewidth=2,
                markersize=5,
                color=color,
                label=algo,
            )

        title = "F1 Score" if metric == "f1" else metric.title()
        ax.set_title(title)
        ax.set_xlabel("Proportion of Duplicates in Dataset")
        ax.set_ylabel("Score")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_ylim(0, 1.0)

        legend = ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=True,
            fontsize=9,
            title="Algorithm",
            title_fontsize=9.5,
            borderpad=0.8,
            labelspacing=0.6,
            handlelength=2.0,
            handleheight=1.0,
        )
        legend.get_frame().set_linewidth(0.8)
        legend.get_frame().set_edgecolor("#cccccc")
        legend.get_frame().set_facecolor("#fafafa")
        legend.get_title().set_fontweight("bold")

        out = output_path.with_name(f"{output_path.stem}_{metric}{output_path.suffix}")
        fig.tight_layout(rect=(0, 0, 0.82, 1.0))
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved plot to {out}")

        csv_out = output_path.with_name(f"{output_path.stem}_{metric}.csv")
        write_metric_csv(csv_out, p_values, algos, rows, metric)
        print(f"Saved CSV to {csv_out}")


if __name__ == "__main__":
    main()