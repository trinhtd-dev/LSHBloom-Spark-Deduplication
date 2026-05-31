import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ đo hiệu năng scale (Thời gian chạy, RAM, Truy vấn) từ scale_results.csv.")
    parser.add_argument(
        "--csv-path", 
        default="outputs/runs/scale_runs_percent/scale_results.csv", 
        help="Đường dẫn đến file CSV chứa kết quả scale"
    )
    parser.add_argument(
        "--out-dir", 
        default="outputs/plots/scale_runs_percent", 
        help="Thư mục lưu biểu đồ kết quả"
    )
    return parser.parse_args()

args = get_args()
csv_file = Path(args.csv_path)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

if not csv_file.exists():
    print(f"[ERROR] Không tìm thấy file CSV tại: {csv_file}")
    exit(1)

# Đọc dữ liệu scale
df = pd.read_csv(csv_file)

# Chuẩn hóa tên thuật toán để khớp cấu hình thống nhất
# 'minhash_lsh' -> 'lsh'
# 'lsh_bloom' -> 'lsh_bloom'
# 'lsh_blowchoc' -> 'lsh_blowchoc'
df['algo_clean'] = df['algo_name'].replace({
    'minhash_lsh': 'lsh',
    'lsh_bloom': 'lsh_bloom',
    'lsh_blowchoc': 'lsh_blowchoc'
})

# Sắp xếp theo n_docs tăng dần để vẽ đường liên tục
df = df.sort_values(by="n_docs")

# Cấu hình đường nét, màu sắc thống nhất giống biểu đồ chất lượng
algos_config = {
    "lsh": {
        "label": "Standard MinHash LSH (Gốc)",
        "color": "#1f77b4", # Xanh dương
        "marker": "s",      # Hình vuông
        "linestyle": "--"   # Nét đứt (Dashed)
    },
    "lsh_bloom": {
        "label": "LSH Bloom (Bộ lọc Bloom)",
        "color": "#9467bd", # Tím
        "marker": "o",      # Hình tròn
        "linestyle": "-."   # Nét chấm gạch (Dash-dot)
    },
    "lsh_blowchoc": {
        "label": "LSH BlowChoc (Cải tiến tối ưu)",
        "color": "#2ca02c", # Xanh lá
        "marker": "^",      # Hình tam giác
        "linestyle": "-"    # Nét liền (Solid)
    }
}

# Lọc danh sách n_docs thực tế có trong dữ liệu (ví dụ: 200k, 400k, ...)
n_docs_list = sorted(df["n_docs"].unique())
x_labels = [f"{int(x/1000)}k" for x in n_docs_list]

# Helper vẽ biểu đồ riêng lẻ
def plot_metric(metric_col, ylabel, title, filename, is_time=False):
    plt.figure(figsize=(8.5, 5.5))
    
    for algo, config in algos_config.items():
        algo_df = df[df["algo_clean"] == algo]
        if algo_df.empty:
            continue
        
        ys = algo_df[metric_col].tolist()
        xs = algo_df["n_docs"].tolist()
        
        # Nếu là thời gian, có thể đổi sang phút nếu số lượng quá lớn (chỉ khi vẽ wall_clock và vượt quá 1000s)
        plot_ys = ys
        plot_ylabel = ylabel
        if is_time and max(ys) > 1000:
            plot_ys = [y / 60.0 for y in ys]
            plot_ylabel = ylabel + " (Phút / Minutes)"
            
        plt.plot(
            xs, 
            plot_ys, 
            label=config["label"], 
            color=config["color"], 
            marker=config["marker"], 
            linestyle=config["linestyle"],
            linewidth=2.5,
            markersize=8,
            alpha=0.95
        )
    
    plt.title(title, fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Số lượng văn bản (Number of Documents)", fontsize=11, labelpad=8)
    plt.ylabel(plot_ylabel, fontsize=11, labelpad=8)
    plt.xticks(n_docs_list, x_labels)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=10.5, frameon=True)
    plt.tight_layout()
    
    save_path = out_dir / filename
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"-> Đã vẽ xong biểu đồ: {save_path.resolve()}")

# 1. Vẽ tổng thời gian chạy (Wall-clock time)
plot_metric(
    metric_col="wall_clock_sec",
    ylabel="Thời gian (Seconds)",
    title="So sánh Tổng Thời Gian Chạy (Wall-Clock Time) khi Scale",
    filename="scale_wall_clock.png",
    is_time=True
)

# 2. Vẽ bộ nhớ RAM đỉnh (Peak RAM Usage)
plot_metric(
    metric_col="peak_ram_gb",
    ylabel="Peak RAM (GB)",
    title="So sánh Tiêu Thụ RAM Đỉnh (Peak RAM Usage) khi Scale",
    filename="scale_ram_peak.png"
)

# 3. Vẽ thời gian truy vấn tìm trùng lặp (Query Time)
plot_metric(
    metric_col="query_sec",
    ylabel="Thời gian truy vấn (Seconds)",
    title="So sánh Thời Gian Truy Vấn Trùng Lặp (Query Time) khi Scale",
    filename="scale_query_time.png"
)

# 4. Vẽ thời gian xây dựng chữ ký (Signature Building Time)
plot_metric(
    metric_col="build_signature_sec",
    ylabel="Thời gian xây dựng (Seconds)",
    title="So sánh Thời Gian Xây Dựng Chữ Ký MinHash khi Scale",
    filename="scale_signature_build.png"
)

# --- 5. Vẽ biểu đồ ghép chất lượng cao gồm 2 đồ thị con (Thời gian chạy & Peak RAM) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

# Đồ thị con bên trái: Tổng thời gian chạy (đổi sang phút cho dễ nhìn)
ax0 = axes[0]
for algo, config in algos_config.items():
    algo_df = df[df["algo_clean"] == algo]
    if algo_df.empty:
        continue
    xs = algo_df["n_docs"].tolist()
    ys = [y / 60.0 for y in algo_df["wall_clock_sec"].tolist()] # Sang phút
    
    ax0.plot(
        xs, ys, 
        label=config["label"], 
        color=config["color"], 
        marker=config["marker"], 
        linestyle=config["linestyle"],
        linewidth=2.5,
        markersize=7,
        alpha=0.95
    )
ax0.set_title("Tổng Thời Gian Chạy (Wall-Clock Time)", fontsize=12, fontweight="bold", pad=12)
ax0.set_xlabel("Số lượng văn bản (Number of Documents)", fontsize=10)
ax0.set_ylabel("Thời gian (Phút / Minutes)", fontsize=10)
ax0.set_xticks(n_docs_list)
ax0.set_xticklabels(x_labels)
ax0.grid(True, linestyle="--", alpha=0.5)

# Đồ thị con bên phải: Tiêu thụ RAM đỉnh
ax1 = axes[1]
for algo, config in algos_config.items():
    algo_df = df[df["algo_clean"] == algo]
    if algo_df.empty:
        continue
    xs = algo_df["n_docs"].tolist()
    ys = algo_df["peak_ram_gb"].tolist()
    
    ax1.plot(
        xs, ys, 
        label=config["label"], 
        color=config["color"], 
        marker=config["marker"], 
        linestyle=config["linestyle"],
        linewidth=2.5,
        markersize=7,
        alpha=0.95
    )
ax1.set_title("Tiêu Thụ RAM Đỉnh (Peak RAM Usage)", fontsize=12, fontweight="bold", pad=12)
ax1.set_xlabel("Số lượng văn bản (Number of Documents)", fontsize=10)
ax1.set_ylabel("RAM peak (GB)", fontsize=10)
ax1.set_xticks(n_docs_list)
ax1.set_xticklabels(x_labels)
ax1.grid(True, linestyle="--", alpha=0.5)

# Đặt chú thích (Legend) chung ở trên cùng
handles, labels = ax0.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=3, fontsize=11, frameon=True)

# Tinh chỉnh khoảng cách giữa các đồ thị con
plt.tight_layout(rect=[0, 0, 1, 0.88])

save_path_combined = out_dir / "scale_performance_combined.png"
plt.savefig(save_path_combined, dpi=200)
plt.close()
print(f"-> Đã vẽ xong biểu đồ ghép: {save_path_combined.resolve()}")

print(f"\n[✓] Vẽ toàn bộ biểu đồ so sánh hiệu năng scale thành công!")
