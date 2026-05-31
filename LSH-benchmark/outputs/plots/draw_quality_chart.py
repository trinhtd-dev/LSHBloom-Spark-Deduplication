import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ chất lượng LSH (Precision, Recall, F1) từ kết quả CSV.")
    parser.add_argument(
        "--dataset", 
        default="test_p_0.5", 
        help="Tên bộ dữ liệu test muốn vẽ (ví dụ: test_p_0.1, test_p_0.5...)"
    )
    return parser.parse_args()

args = get_args()

# Thư mục chứa kết quả dựa trên tham số truyền vào
results_root = Path("outputs/runs") / args.dataset
out_dir = Path("outputs/plots") / args.dataset
out_dir.mkdir(parents=True, exist_ok=True)
# Các thuật toán cần vẽ và cấu hình đường biểu diễn
algos = {
    "lsh": {
        "dir": "lsh_results",
        "prefix": "lsh",
        "label": "Standard MinHash LSH (Gốc)",
        "color": "#1f77b4", # Xanh dương
        "marker": "s",      # Hình vuông
        "linestyle": "--"   # Nét đứt (Dashed)
    },
    "lsh_bloom": {
        "dir": "lsh_bloom_results",
        "prefix": "lsh_bloom",
        "label": "LSH Bloom (Bộ lọc Bloom)",
        "color": "#9467bd", # Tím
        "marker": "o",      # Hình tròn
        "linestyle": "-."   # Nét chấm gạch (Dash-dot)
    },
    "lsh_blowchoc": {
        "dir": "lsh_blowchoc_results",
        "prefix": "lsh_blowchoc",
        "label": "LSH BlowChoc (Word-Aligned)",
        "color": "#2ca02c", # Xanh lá
        "marker": "^",      # Hình tam giác
        "linestyle": "-"    # Nét liền (Solid)
    },
    "lsh_blowchoc_choices": {
        "dir": "lsh_blowchoc_choices_results",
        "prefix": "lsh_blowchoc_choices",
        "label": "LSH BlowChoc (Choices - Chuẩn)",
        "color": "#ff7f0e", # Cam
        "marker": "D",      # Hình kim cương
        "linestyle": ":"    # Nét đứt chấm (Dotted)
    }
}

# Khởi tạo tập hợp chứa tất cả các ngưỡng tìm thấy trong thực tế
found_thresholds = set()
for algo, config in algos.items():
    algo_dir = results_root / config["dir"]
    if algo_dir.exists():
        # Tìm các file score dạng {prefix}_{threshold}_{num_perm}_score.csv hoặc {prefix}_{threshold}_score.csv
        for p in algo_dir.glob(f"{config['prefix']}_*_score.csv"):
            name = p.name
            # Loại bỏ prefix và _score.csv
            part = name[len(config['prefix'])+1 : -len("_score.csv")]
            if "_" in part:
                t_str = part.split("_")[0]
            else:
                t_str = part
            try:
                found_thresholds.add(float(t_str))
            except ValueError:
                pass

if found_thresholds:
    thresholds = sorted(list(found_thresholds))
    print(f"-> Tìm thấy các ngưỡng thực tế: {thresholds}")
else:
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"-> Không tìm thấy file kết quả nào, sử dụng danh sách ngưỡng mặc định: {thresholds}")

# Đọc dữ liệu từ các file CSV
data = {algo: {"precision": [], "recall": [], "f1": []} for algo in algos}

for algo, config in algos.items():
    algo_dir = results_root / config["dir"]
    for t in thresholds:
        score_file = None
        # Thử các đường dẫn khả dĩ với các cách định dạng chuỗi số thực khác nhau
        t_formats = [f"{t:.2f}", f"{t:.1f}", str(t)]
        # Chuẩn hóa để loại bỏ số 0 vô nghĩa ở đuôi thập phân (ví dụ 0.50 -> 0.5)
        t_formats_clean = []
        for tf in t_formats:
            if "." in tf:
                tf_clean = tf.rstrip('0').rstrip('.')
                if not tf_clean:
                    tf_clean = "0"
                t_formats_clean.append(tf_clean)
            t_formats_clean.append(tf)
        
        # Tạo danh sách các đường dẫn cần thử
        paths_to_try = []
        for tf in sorted(set(t_formats_clean), key=len, reverse=True):
            paths_to_try.extend([
                algo_dir / f"{config['prefix']}_{tf}_128_score.csv",
                algo_dir / f"{config['prefix']}_{tf}_score.csv"
            ])
        # Đảm bảo thử thêm chính xác định dạng floating-point trực tiếp
        paths_to_try.extend([
            algo_dir / f"{config['prefix']}_{t}_128_score.csv",
            algo_dir / f"{config['prefix']}_{t}_score.csv"
        ])
        
        for path in paths_to_try:
            if path.exists():
                score_file = path
                break
        
        if score_file and score_file.exists():
            df = pd.read_csv(score_file)
            data[algo]["precision"].append(df["precision"].iloc[0])
            data[algo]["recall"].append(df["recall"].iloc[0])
            data[algo]["f1"].append(df["f1"].iloc[0])
        else:
            print(f"Thiếu file dữ liệu cho ngưỡng {t} của thuật toán {algo} (Thử tìm trong: {[p.name for p in paths_to_try[:4]]})")
            data[algo]["precision"].append(None)
            data[algo]["recall"].append(None)
            data[algo]["f1"].append(None)

# Định nghĩa các thông số đo lường và tiêu đề biểu đồ
metrics = ["precision", "recall", "f1"]
titles = ["Độ chính xác (Precision) vs Threshold", "Độ bao phủ (Recall) vs Threshold", "Chỉ số F1-Score vs Threshold"]
ylabels = ["Precision", "Recall", "F1 Score"]

# --- Định nghĩa hàm Jitter (dịch chuyển nhẹ trục X để tránh trùng lặp điểm marker) ---
def apply_jitter(xs, algo):
    if algo == "lsh":
        return [x - 0.012 for x in xs] # Dịch trái một chút
    elif algo == "lsh_bloom":
        return [x for x in xs]         # Giữ nguyên ở giữa
    elif algo == "lsh_blowchoc":
        return [x + 0.012 for x in xs] # Dịch phải một chút
    return xs

# --- Vẽ 3 Biểu đồ riêng lẻ (Tách riêng biệt theo yêu cầu - Đúng số thật 100% không dịch chuyển) ---
for idx, metric in enumerate(metrics):
    plt.figure(figsize=(8, 5))
    
    all_ys = []
    for algo, config in algos.items():
        ys = data[algo][metric]
        valid_ys = [y for y in ys if y is not None]
        if valid_ys:
            all_ys.extend(valid_ys)
            
        plt.plot(
            thresholds, 
            ys, 
            label=config["label"], 
            color=config["color"], 
            marker=config["marker"], 
            linestyle=config["linestyle"],
            linewidth=2,
            markersize=8,
            alpha=0.9
        )
    
    plt.title(titles[idx], fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Jaccard Similarity Threshold", fontsize=11)
    plt.ylabel(ylabels[idx], fontsize=11)
    
    # Tự động điều chỉnh khoảng zoom của trục Y (Zoom Y-Axis) để kéo dãn đồ thị
    if all_ys:
        min_y = min(all_ys)
        max_y = max(all_ys)
        y_margin = max(0.04, (max_y - min_y) * 0.15)
        plt.ylim(max(-0.02, min_y - y_margin), min(1.02, max_y + y_margin))
    else:
        plt.ylim(-0.05, 1.05)
        
    plt.xticks(thresholds)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", fontsize=10, frameon=True)
    plt.tight_layout()
    
    # Lưu biểu đồ riêng lẻ
    single_save_path = out_dir / f"{metric}_comparison.png"
    plt.savefig(single_save_path, dpi=200)
    plt.close()
    print(f"-> Đã vẽ xong biểu đồ riêng lẻ (Đúng tọa độ gốc 100%): {single_save_path.resolve()}")

# --- Vẽ biểu đồ ghép gồm 3 đồ thị con nằm ngang (Bản backup) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for idx, metric in enumerate(metrics):
    ax = axes[idx]
    all_ys = []
    
    for algo, config in algos.items():
        ys = data[algo][metric]
        valid_ys = [y for y in ys if y is not None]
        if valid_ys:
            all_ys.extend(valid_ys)
            
        ax.plot(
            thresholds, 
            ys, 
            label=config["label"], 
            color=config["color"], 
            marker=config["marker"], 
            linestyle=config["linestyle"],
            linewidth=2,
            markersize=7,
            alpha=0.9
        )
    ax.set_title(titles[idx], fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Jaccard Similarity Threshold", fontsize=10)
    ax.set_ylabel(ylabels[idx], fontsize=10)
    
    if all_ys:
        min_y = min(all_ys)
        max_y = max(all_ys)
        y_margin = max(0.04, (max_y - min_y) * 0.15)
        ax.set_ylim(max(-0.02, min_y - y_margin), min(1.02, max_y + y_margin))
    else:
        ax.set_ylim(-0.05, 1.05)
        
    ax.set_xticks(thresholds)
    ax.grid(True, linestyle="--", alpha=0.5)

# Đặt chú thích (Legend) chung ở trên cùng
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=3, fontsize=11, frameon=True)

# Tinh chỉnh khoảng cách giữa các đồ thị con
plt.tight_layout(rect=[0, 0, 1, 0.88])

# Lưu biểu đồ ghép
save_path = out_dir / "quality_comparison_combined.png"
plt.savefig(save_path, dpi=200)
plt.close()

print(f"\n[✓] Vẽ toàn bộ biểu đồ chuẩn xác 100% tọa độ gốc thành công!")





