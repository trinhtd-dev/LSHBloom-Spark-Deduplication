from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = BASE_DIR / "psweep_summary.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "psweep_compare_all.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all algorithms comparison at one dataset + threshold."
    )
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--dataset", required=True, help="e.g. test_p_0.5")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--metric",
        default="f1",
        help="precision, recall, f1, auc_roc, acc, bal_acc",
    )
    parser.add_argument(
        "--kind",
        choices=["bar", "line"],
        default="line",
        help="bar = compare algos at one threshold; line = metric vs threshold by algo",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    rows = load_rows(summary_path)

    dataset = args.dataset.strip()
    metric = args.metric.strip()

    rows = [r for r in rows if r.get("dataset") == dataset]
    if not rows:
        raise SystemExit(f"No rows found for dataset={dataset}")

    if metric not in rows[0]:
        raise SystemExit(f"Metric '{metric}' not found in summary CSV")

    if args.kind == "bar":
        threshold = float(f"{args.threshold:.1f}")
        rows = [r for r in rows if float(r.get("threshold", 0)) == threshold]
        if not rows:
            raise SystemExit(
                f"No rows found for dataset={dataset} threshold={threshold:.1f}"
            )

        rows.sort(key=lambda r: float(r.get(metric, 0) or 0), reverse=True)
        algos = [r["algorithm"] for r in rows]
        values = [float(r.get(metric, 0) or 0) for r in rows]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(algos, values, color="#4c72b0")
        ax.set_title(f"{metric} comparison @ {dataset}, threshold={threshold:.1f}")
        ax.set_xlabel("algorithm")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)
    else:
        thresholds = sorted({float(r.get("threshold", 0)) for r in rows})
        algos = sorted({r.get("algorithm", "") for r in rows})
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in algos:
            points = [r for r in rows if r.get("algorithm") == algo]
            points.sort(key=lambda r: float(r.get("threshold", 0)))
            xs = [float(r.get("threshold", 0)) for r in points]
            ys = [float(r.get(metric, 0) or 0) for r in points]
            ax.plot(xs, ys, marker="o", label=algo)

        ax.set_title(f"{metric} vs threshold @ {dataset}")
        ax.set_xlabel("threshold")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, ncol=2, title="algorithm")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
