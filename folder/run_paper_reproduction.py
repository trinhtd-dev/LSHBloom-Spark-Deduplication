import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold for LSH")
    parser.add_argument("--ngram", type=int, default=1, help="N-Gram size for MinHashing")
    parser.add_argument("--num-perm", type=int, default=128, help="Number of hash functions for MinHashing")
    return parser.parse_args()

def main():
    args = get_args()
    p_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    f1_scores_bloom = []
    f1_scores_lsh = []


    for p in p_values:
        benchmark_name = f"test_p_{p}"
        print(f"\n{'='*20} RUNNING FOR p = {p} {'='*20}")
        
        # 1. Tạo tập dữ liệu Test với tỷ lệ p tương ứng
        print(f"1. Generating dataset {benchmark_name} (Duplicate proportion: {p})...")
        subprocess.run([
            "python", "synthetic_benchmark/create_lshbloom_benchmark.py", 
            "-p", str(p), 
            "-o", benchmark_name
        ], check=True)
        
        # 2. Chạy thuật toán LSH Bloom
        print(f"2. Running LSHBloom deduplication for {benchmark_name} with Threshold={args.threshold}, N-Gram={args.ngram}, Perms={args.num_perm}...")
        subprocess.run([
            "python", "dedup/lsh/lsh_bloom.py", 
            "--input", benchmark_name,
            "--sim-threshold", str(args.threshold),
            "--ngram", str(args.ngram),
            "--num-perm", str(args.num_perm)
        ], check=True)
        
        # 3. Trích xuất điểm F1 từ file kết quả LSH Bloom
        score_file_bloom = f"{benchmark_name}/lsh_bloom_results/lsh_bloom_{args.threshold}_{args.num_perm}_score.csv"
        if os.path.exists(score_file_bloom):
            df = pd.read_csv(score_file_bloom)
            f1 = df['f1'].iloc[0]
            f1_scores_bloom.append(f1)
            print(f"-> FINISHED LSHBloom p={p}: F1 Score = {f1:.4f}")
        else:
            print(f"-> ERROR: Score file {score_file_bloom} not found!")
            f1_scores_bloom.append(None)
            
        # 4. Chạy thuật toán LSH Chuẩn (với Redis)
        print(f"4. Running MinHashLSH deduplication for {benchmark_name} with Threshold={args.threshold}, N-Gram={args.ngram}, Perms={args.num_perm}...")
        subprocess.run([
            "python", "dedup/lsh/lsh.py", 
            "--input", benchmark_name,
            "--sim-threshold", str(args.threshold),
            "--ngram", str(args.ngram),
            "--num-perm", str(args.num_perm)
        ], check=True)
        
        # 5. Trích xuất điểm F1 từ file kết quả LSH Chuẩn
        score_file_lsh = f"{benchmark_name}/lsh_results/lsh_{args.threshold}_{args.num_perm}_score.csv"
        if os.path.exists(score_file_lsh):
            df_lsh = pd.read_csv(score_file_lsh)
            f1_lsh = df_lsh['f1'].iloc[0]
            f1_scores_lsh.append(f1_lsh)
            print(f"-> FINISHED MinHashLSH p={p}: F1 Score = {f1_lsh:.4f}")
        else:
            print(f"-> ERROR: Score file {score_file_lsh} not found!")
            f1_scores_lsh.append(None)

    # 6. Vẽ lại Đồ thị so sánh 2 phương pháp
    print("\n" + "="*50)
    print("All iterations complete! Plotting results...")
    
    plt.figure(figsize=(10, 6))
    
    # Đường LSH Bloom (màu tím)
    plt.plot(p_values, f1_scores_bloom, marker='o', linestyle='-', color='orchid', label='LSHBloom')
    
    # Đường LSH tiêu chuẩn (màu xanh blue)
    plt.plot(p_values, f1_scores_lsh, marker='s', linestyle='--', color='royalblue', label='MinHashLSH (Redis)')
    
    plt.xlabel('Proportion of Positive Labels', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.title(f'Comparison: LSHBloom vs MinHashLSH (Threshold={args.threshold}, Perm={args.num_perm}, N-Gram={args.ngram})', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(p_values, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0.2, 1.0)
    
    # Save ra file ảnh
    save_path = 'reproduction_plot.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"SUCCESS: Plot saved to {save_path}!")

if __name__ == "__main__":
    main()