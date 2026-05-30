import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot peS2o scale results from scale_results.csv.")
    parser.add_argument(
        "--csv",
        default=str(Path("scale_runs") / "scale_results.csv"),
        help="Path to scale_results.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path("visualize") / "pes2o_scale_plots"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--algos",
        default="",
        help="Comma-separated algos to plot (empty = all in CSV)",
    )
    return parser.parse_args()


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    for algo, group in df.groupby("algo_name"):
        group = group.sort_values("n_docs")
        plt.plot(group["n_docs"], group[metric], marker="o", label=algo)
    plt.xlabel("Number of documents")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs number of documents")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scatter(
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(7, 4))
    for algo, group in df.groupby("algo_name"):
        plt.scatter(group[x_metric], group[y_metric], label=algo, alpha=0.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs {xlabel}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "algo_name" not in df.columns:
        raise ValueError("CSV missing 'algo_name' column")

    if args.algos.strip():
        selected = {a.strip() for a in args.algos.split(",") if a.strip()}
        df = df[df["algo_name"].isin(selected)]

    out_dir = Path(args.out_dir)

    plot_metric(df, "wall_clock_sec", "Wall-clock time (sec)", out_dir / "runtime.png")
    plot_metric(df, "disk_usage_gb", "Disk usage (GB)", out_dir / "disk_usage.png")
    if "peak_ram_gb" in df.columns:
        plot_metric(df, "peak_ram_gb", "Peak RAM (GB)", out_dir / "peak_ram.png")
        plot_scatter(
            df,
            "wall_clock_sec",
            "peak_ram_gb",
            "Wall-clock time (sec)",
            "Peak RAM (GB)",
            out_dir / "scatter_runtime_vs_ram.png",
        )
    plot_scatter(
        df,
        "wall_clock_sec",
        "disk_usage_gb",
        "Wall-clock time (sec)",
        "Disk usage (GB)",
        out_dir / "scatter_runtime_vs_disk.png",
    )

    print(f"Saved plots to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
