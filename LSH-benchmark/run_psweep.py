from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

# Adjust if you only want one algorithm.
ALGORITHMS = [
    (
        "lsh",
        ["dedup/lsh/lsh.py", "--num-perm", "128", "--ngram", "1"],
        "lsh_results",
        "lsh",
    ),
    (
        "lsh_bloom",
        ["dedup/lsh/lsh_bloom.py", "--num-perm", "128", "--ngram", "1"],
        "lsh_bloom_results",
        "lsh_bloom",
    ),
    (
        "lsh_oph_doph",
        [
            "dedup/lsh/lsh_oph_doph.py",
            "--num-perm",
            "128",
            "--oph-bins",
            "128",
            "--ngram",
            "1",
        ],
        "lsh_oph_doph_results",
        "lsh_oph_doph",
    ),
    (
        "lsh_multiprobe",
        [
            "dedup/lsh/lsh_multiprobe.py",
            "--num-perm",
            "128",
            "--ngram",
            "1",
            "--num-probes",
            "8",
        ],
        "lsh_multiprobe_results",
        "lsh_multiprobe",
    ),
]

OUTPUT_SUMMARY = BASE_DIR / "psweep_summary.csv"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="f1", help="Metric to plot from summary CSV.")
    parser.add_argument("--plot", action="store_true", help="Generate a plot after running sweep.")
    parser.add_argument(
        "--plot-kind",
        choices=["line", "heatmap", "surface3d"],
        default="line",
        help="Plot type for the sweep summary.",
    )
    parser.add_argument(
        "--plot-out",
        default=str(BASE_DIR / "plots" / "psweep_plot.png"),
        help="Output image path for plot.",
    )
    parser.add_argument(
        "--plot-algos",
        default="",
        help="Comma-separated algorithms to include in plots (e.g., lsh,lsh_bloom).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Plot from existing summary CSV without running sweep.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel jobs to run.",
    )
    parser.add_argument(
        "--algos",
        default="",
        help="Comma-separated algorithms to run (e.g., lsh_oph_doph,lsh_multiprobe).",
    )
    return parser.parse_args()


def dataset_tag(p: float) -> str:
    return f"test_p_{p:.1f}"


def score_file(dataset: str, results_dir: str, algo_prefix: str, threshold: float, num_perm: int) -> Path:
    score_name = f"{algo_prefix}_{threshold:.1f}_{num_perm}_score.csv"
    return BASE_DIR / dataset / results_dir / score_name


def run_cmd(cmd: list[str]) -> int:
    print("\n==>", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=BASE_DIR)
    return proc.returncode


def read_score(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def run_task(task: dict[str, object]) -> dict[str, str] | None:
    tag = str(task["tag"])
    p = float(task["p"])
    threshold = float(task["threshold"])
    algo_name = str(task["algo_name"])
    script_args = list(task["script_args"])
    results_dir = str(task["results_dir"])
    algo_prefix = str(task["algo_prefix"])

    num_perm = int(script_args[script_args.index("--num-perm") + 1])
    cmd = [PYTHON, *script_args, "--sim-threshold", f"{threshold:.1f}", "--input", tag]
    rc = run_cmd(cmd)
    if rc != 0:
        print(f"[error] {algo_name} failed on {tag} @ {threshold:.1f}")
        return None

    sfile = score_file(tag, results_dir, algo_prefix, threshold, num_perm)
    score = read_score(sfile)
    if not score:
        print(f"[warn] score file missing for {algo_name} on {tag}: {sfile}")
        return None

    return {
        "dataset": tag,
        "p": f"{p:.1f}",
        "threshold": f"{threshold:.1f}",
        "algorithm": algo_name,
        "precision": score.get("precision", ""),
        "recall": score.get("recall", ""),
        "f1": score.get("f1", ""),
        "auc_roc": score.get("auc_roc", ""),
        "acc": score.get("acc", ""),
        "bal_acc": score.get("bal_acc", ""),
        "tp": score.get("tp", ""),
        "fp": score.get("fp", ""),
        "fn": score.get("fn", ""),
    }


def plot_summary(summary_path: Path, metric: str, kind: str, output_path: Path, algos_filter: set[str] | None) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    rows = []
    with summary_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if algos_filter:
        rows = [r for r in rows if r.get("algorithm") in algos_filter]

    if not rows:
        print("[warn] summary CSV is empty; skipping plot")
        return

    def label_dataset(name: str) -> str:
        if name.startswith("test_p_"):
            return f"p={name.replace('test_p_', '')}"
        return name

    metric = metric.strip()
    if metric not in rows[0]:
        print(f"[warn] metric '{metric}' not found in summary; skipping plot")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if kind == "surface3d":
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        datasets = sorted({r["dataset"] for r in rows})
        dataset_labels = [label_dataset(d) for d in datasets]
        thresholds = sorted({float(r["threshold"]) for r in rows})
        algos = sorted({r["algorithm"] for r in rows})

        for algo in algos:
            grid = [[0.0 for _ in thresholds] for _ in datasets]
            for r in rows:
                if r["algorithm"] != algo:
                    continue
                d_i = datasets.index(r["dataset"])
                t_i = thresholds.index(float(r["threshold"]))
                grid[d_i][t_i] = float(r.get(metric, 0) or 0)

            xs = np.arange(len(thresholds))
            ys = np.arange(len(datasets))
            X, Y = np.meshgrid(xs, ys)
            Z = np.array(grid)

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
            ax.set_title(f"{algo} {metric} surface")
            ax.set_xlabel("threshold")
            ax.set_ylabel("dataset")
            ax.set_zlabel(metric)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{t:.1f}" for t in thresholds])
            ax.set_yticks(ys)
            ax.set_yticklabels(dataset_labels)
            out = output_path.with_name(f"{output_path.stem}_{algo}{output_path.suffix}")
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
        print(f"Saved plots to {output_path.parent}")
        return

    if kind == "heatmap":
        datasets = sorted({r["dataset"] for r in rows})
        dataset_labels = [label_dataset(d) for d in datasets]
        thresholds = sorted({float(r["threshold"]) for r in rows})
        algos = sorted({r["algorithm"] for r in rows})

        for algo in algos:
            grid = [[None for _ in thresholds] for _ in datasets]
            for r in rows:
                if r["algorithm"] != algo:
                    continue
                d_i = datasets.index(r["dataset"])
                t_i = thresholds.index(float(r["threshold"]))
                grid[d_i][t_i] = float(r.get(metric, 0) or 0)

            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(grid, aspect="auto")
            ax.set_title(f"{algo} {metric} heatmap")
            ax.set_xlabel("threshold")
            ax.set_ylabel("dataset")
            ax.set_xticks(range(len(thresholds)), [f"{t:.1f}" for t in thresholds])
            ax.set_yticks(range(len(datasets)), dataset_labels)
            fig.colorbar(im, ax=ax)
            out = output_path.with_name(f"{output_path.stem}_{algo}{output_path.suffix}")
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
        print(f"Saved plots to {output_path.parent}")
        return

    # Line plot
    algos = sorted({r["algorithm"] for r in rows})
    for algo in algos:
        fig, ax = plt.subplots(figsize=(10, 6))
        algo_rows = [r for r in rows if r["algorithm"] == algo]
        datasets = sorted({r["dataset"] for r in algo_rows})
        for ds in datasets:
            points = [r for r in algo_rows if r["dataset"] == ds]
            points.sort(key=lambda r: float(r["threshold"]))
            xs = [float(r["threshold"]) for r in points]
            ys = [float(r.get(metric, 0) or 0) for r in points]
            ax.plot(xs, ys, marker="o", label=label_dataset(ds))

        ax.set_title(f"{algo} {metric} vs threshold")
        ax.set_xlabel("threshold")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        out = output_path.with_name(f"{output_path.stem}_{algo}{output_path.suffix}")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)

    print(f"Saved plots to {output_path.parent}")


def main() -> int:
    args = get_args()
    selected_algos = {a.strip() for a in args.algos.split(",") if a.strip()}
    if selected_algos:
        active_algorithms = [a for a in ALGORITHMS if a[0] in selected_algos]
        missing = sorted(selected_algos - {a[0] for a in active_algorithms})
        if missing:
            print(f"[error] unknown algorithms: {', '.join(missing)}")
            return 1
    else:
        active_algorithms = ALGORITHMS
    if args.plot_only:
        if OUTPUT_SUMMARY.exists():
            algos_filter = {a.strip() for a in args.plot_algos.split(",") if a.strip()}
            if not algos_filter:
                algos_filter = None
            plot_summary(OUTPUT_SUMMARY, args.metric, args.plot_kind, Path(args.plot_out), algos_filter)
            return 0
        print(f"[error] summary CSV not found: {OUTPUT_SUMMARY}")
        return 1
    rows = []
    thresholds = [i / 10 for i in range(1, 10)]
    tasks: list[dict[str, object]] = []
    for p in [i / 10 for i in range(1, 10)]:
        tag = dataset_tag(p)
        jsonl_path = BASE_DIR / "benchmark_dfs" / f"{tag}.jsonl"
        csv_path = BASE_DIR / "benchmark_dfs" / f"{tag}.csv"

        if not jsonl_path.exists() or not csv_path.exists():
            print(f"[skip] missing benchmark files for {tag}")
            continue

        for threshold in thresholds:
            for algo_name, script_args, results_dir, algo_prefix in active_algorithms:
                tasks.append(
                    {
                        "tag": tag,
                        "p": p,
                        "threshold": threshold,
                        "algo_name": algo_name,
                        "script_args": script_args,
                        "results_dir": results_dir,
                        "algo_prefix": algo_prefix,
                    }
                )

    if args.jobs > 1 and tasks:
        with Pool(processes=args.jobs) as pool:
            for result in pool.imap_unordered(run_task, tasks):
                if result:
                    rows.append(result)
    else:
        for task in tasks:
            result = run_task(task)
            if result:
                rows.append(result)

    if rows:
        with OUTPUT_SUMMARY.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote summary to {OUTPUT_SUMMARY}")
        if args.plot:
            algos_filter = {a.strip() for a in args.plot_algos.split(",") if a.strip()}
            if not algos_filter:
                algos_filter = None
            plot_summary(OUTPUT_SUMMARY, args.metric, args.plot_kind, Path(args.plot_out), algos_filter)
    else:
        print("\nNo results collected.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
