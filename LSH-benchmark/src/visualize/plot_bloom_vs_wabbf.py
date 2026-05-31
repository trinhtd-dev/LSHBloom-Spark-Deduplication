import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Setup styles for a premium look
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'font.family': 'sans-serif'
})

def create_plots():
    results_path = Path("scale_runs_2M/scale_results.csv")
    if not results_path.exists():
        results_path = Path("scale_runs/scale_results.csv")
    
    if not results_path.exists():
        print(f"Error: No benchmark results found at {results_path}")
        return
        
    print(f"Loading 2M scale benchmark results from: {results_path}")
    df = pd.read_csv(results_path)
    
    # Filter algos
    target_algos = ["lsh_bloom", "lsh_wabbf", "lsh_blocked_bloom"]
    df_plot = df[df["algo_name"].isin(target_algos)].copy()
    
    label_map = {
        "lsh_bloom": "LSH Bloom (Standard)",
        "lsh_wabbf": "LSH WA-BBF (Word-Aligned)",
        "lsh_blocked_bloom": "LSH Blocked Bloom"
    }
    df_plot["label"] = df_plot["algo_name"].map(label_map)
    df_plot = df_plot.sort_values(["algo_name", "n_docs"])
    
    # Convert doc counts to millions for readable labels (e.g., 0.25M, 1.0M)
    df_plot["n_docs_million"] = df_plot["n_docs"] / 1_000_000.0
    
    out_dir = Path("visualize/bloom_vs_wabbf_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    colors = {
        "LSH Bloom (Standard)": "#4A90E2",
        "LSH WA-BBF (Word-Aligned)": "#E25A90",
        "LSH Blocked Bloom": "#50E3C2"
    }
    
    # --- PLOT 1: Wall-Clock Total Runtime Scaling ---
    plt.figure(figsize=(8, 5))
    for label, group in df_plot.groupby("label"):
        color = colors.get(label, "#333333")
        plt.plot(group["n_docs_million"], group["wall_clock_sec"], marker="o", linewidth=2.5, markersize=8, label=label, color=color)
        
        # Add labels to points
        for x, y in zip(group["n_docs_million"], group["wall_clock_sec"]):
            plt.text(x, y + 25, f"{int(y)}s", ha="center", fontsize=9, fontweight="bold")

    plt.xlabel("Quy mô dữ liệu (Triệu dòng - Documents in Millions)")
    plt.ylabel("Tổng thời gian thực thi (giây)")
    plt.title("So sánh Tổng thời gian khử trùng lặp (Wall-Clock Time)\nScale từ 250K đến 2M dòng peS2o", pad=15)
    plt.xticks(df_plot["n_docs_million"].unique())
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "total_runtime_scaling.png", dpi=150)
    plt.close()
    
    # --- PLOT 2: Query Only Scaling ---
    plt.figure(figsize=(8, 5))
    for label, group in df_plot.groupby("label"):
        color = colors.get(label, "#333333")
        plt.plot(group["n_docs_million"], group["query_sec"], marker="s", linewidth=2.5, markersize=8, label=label, color=color)
        
        # Add labels to points
        for x, y in zip(group["n_docs_million"], group["query_sec"]):
            plt.text(x, y + 0.8, f"{y:.2f}s", ha="center", fontsize=9, fontweight="bold")

    plt.xlabel("Quy mô dữ liệu (Triệu dòng - Documents in Millions)")
    plt.ylabel("Thời gian truy vấn bộ lọc (giây)")
    plt.title("So sánh tốc độ truy vấn bộ lọc (Filter Query Time)\nScale từ 250K đến 2M dòng peS2o", pad=15)
    plt.xticks(df_plot["n_docs_million"].unique())
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "query_time_scaling.png", dpi=150)
    plt.close()
    
    # --- PLOT 3: Peak RAM & Disk Footprints ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for label, group in df_plot.groupby("label"):
        color = colors.get(label, "#333333")
        ax1.plot(group["n_docs_million"], group["peak_ram_gb"], marker="^", linewidth=2, markersize=7, label=label, color=color)
        ax2.plot(group["n_docs_million"], group["disk_usage_gb"], marker="v", linewidth=2, markersize=7, label=label, color=color)

    ax1.set_xlabel("Quy mô dữ liệu (Triệu dòng)")
    ax1.set_ylabel("Peak RAM Usage (GB)")
    ax1.set_title("RAM tiêu thụ tối đa (Peak RAM)")
    ax1.set_xticks(df_plot["n_docs_million"].unique())
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()
    
    ax2.set_xlabel("Quy mô dữ liệu (Triệu dòng)")
    ax2.set_ylabel("Disk Usage (GB)")
    ax2.set_title("Dung lượng bộ lọc lưu trên đĩa (Disk Usage)")
    ax2.set_xticks(df_plot["n_docs_million"].unique())
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    
    plt.suptitle("So sánh tài nguyên hệ thống (RAM & Disk Usage) khi scaling", y=0.98, fontsize=15)
    plt.tight_layout()
    plt.savefig(out_dir / "resource_scaling.png", dpi=150)
    plt.close()
    
    print(f"Successfully generated scale plots in: {out_dir.resolve()}")

if __name__ == "__main__":
    create_plots()
