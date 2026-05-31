import streamlit as st
import sys
import os
import time
import re
import shutil
from pathlib import Path
import xxhash
import numpy as np

# Thiết lập đường dẫn import đến các thư mục thuật toán
CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CURRENT_DIR.parent
LSH_DIR = WORKSPACE_DIR / "src" / "dedup" / "lsh"

# Phải chèn thư mục wrapper datasketch để Python tìm thấy package 'datasketch' custom
if str(LSH_DIR / "datasketch") not in sys.path:
    sys.path.insert(0, str(LSH_DIR / "datasketch"))
if str(LSH_DIR) not in sys.path:
    sys.path.insert(0, str(LSH_DIR))
if str(WORKSPACE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR / "src"))

# Import các thuật toán băm và LSH
from datasketch import MinHash, MinHashLSH, MinHashLSHBloom
from lsh_blowchoc import MinHashLSHBlowChoc


# Tiền xử lý văn bản đơn giản giống trong benchmark
TOKEN_RE = re.compile(r"\w+")

def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())

def build_minhash(text: str, num_perm: int, ngram_n: int):
    tokens = tokenize(text)
    mh = MinHash(num_perm=num_perm)
    
    if not tokens:
        mh.update(b"__EMPTY__")
        return mh
        
    if len(tokens) < ngram_n:
        sh = " ".join(tokens).encode("utf-8", errors="ignore")
        mh.update(sh)
    else:
        seen = set()
        for i in range(len(tokens) - ngram_n + 1):
            sh = " ".join(tokens[i : i + ngram_n]).encode("utf-8", errors="ignore")
            h = xxhash.xxh64_digest(sh)
            if h not in seen:
                seen.add(h)
                mh.update(sh) # Sử dụng shingle encode để update minhash
    return mh

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="LSH Deduplication Interactive Demo",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Trực quan hóa Phát hiện Trùng lặp Văn bản bằng LSH")
st.markdown("""
Ứng dụng này giúp bạn kiểm tra khả năng phát hiện trùng lặp (Deduplication) của các thuật toán LSH: **Standard MinHash LSH**, **LSH Bloom**, và **LSH BlowChoc** trên dữ liệu thực tế.
""")

# Tạo thư mục tạm để lưu trạng thái bộ lọc Bloom/BlowChoc
temp_save_dir = WORKSPACE_DIR / "outputs" / "demo_temp"
os.makedirs(temp_save_dir, exist_ok=True)

# Khởi tạo dữ liệu mẫu
sample_texts = [
    "Locality Sensitive Hashing (LSH) là một kỹ thuật băm bảo toàn độ tương đồng của dữ liệu văn bản.",
    "Hôm nay trời nắng đẹp, tôi quyết định đi dạo công viên và đọc sách dưới tán cây.",
    "Locality-Sensitive Hashing (LSH) là phương pháp băm giữ nguyên độ tương đồng đối với các văn bản gốc.",
    "Hôm nay thời tiết rất đẹp, tôi quyết định đi dạo ngoài công viên để đọc một cuốn sách hay.",
    "Thuật toán LSH Bloom sử dụng bộ lọc Bloom để tăng tốc độ truy vấn trùng lặp lên gấp nhiều lần.",
    "Mô hình học máy cần lượng dữ liệu lớn và sạch để đạt hiệu năng tối ưu nhất trong thực tế.",
    "Thuật toán LSH Bloom tích hợp bộ lọc Bloom giúp tăng tốc độ tìm kiếm các văn bản bị trùng lặp.",
    "Dự án nghiên cứu này tập trung tối ưu hóa cấu trúc bộ nhớ cache cho các bộ lọc Bloom nâng cao."
]

# Sidebar cấu hình tham số
st.sidebar.header("⚙️ Cấu hình thuật toán")

algo_choice = st.sidebar.selectbox(
    "Chọn Thuật toán LSH",
    ["Standard MinHash LSH", "LSH Bloom", "LSH BlowChoc"]
)

threshold = st.sidebar.slider(
    "Ngưỡng tương đồng (Jaccard Threshold)",
    min_value=0.1,
    max_value=1.0,
    value=0.6,
    step=0.05,
    help="Độ tương đồng tối thiểu để coi 2 văn bản là trùng lặp"
)

num_perm = st.sidebar.select_slider(
    "Độ dài chữ ký MinHash (num-perm)",
    options=[16, 32, 64, 128, 256],
    value=64,
    help="Số lượng hàm băm. Số lớn hơn cho kết quả chính xác hơn nhưng chậm hơn."
)

ngram_n = st.sidebar.slider(
    "Kích thước N-gram",
    min_value=1,
    max_value=10,
    value=3,
    help="Độ dài cụm từ để băm"
)

# Khu vực chính nhập liệu
st.subheader("📝 Nhập danh sách văn bản cần kiểm tra")
st.caption("Nhập mỗi văn bản/câu trên một dòng. Hệ thống sẽ quét từ trên xuống dưới để phát hiện các câu trùng lặp với câu phía trước.")

# Nút tải dữ liệu mẫu
if st.button("📥 Tải dữ liệu mẫu tiếng Việt"):
    st.session_state["input_area"] = "\n".join(sample_texts)

input_text = st.text_area(
    "Danh sách văn bản:",
    value=st.session_state.get("input_area", ""),
    height=250,
    key="input_area_widget"
)

if st.button("🚀 Bắt đầu khử trùng lặp", type="primary"):
    lines = [line.strip() for line in input_text.split("\n") if line.strip()]
    
    if not lines:
        st.warning("Vui lòng nhập ít nhất một câu văn bản!")
    else:
        st.info(f"Đang phân tích {len(lines)} văn bản bằng thuật toán **{algo_choice}**...")
        
        # Xóa các file lọc cũ trong thư mục tạm
        for f in temp_save_dir.glob("*"):
            if f.is_file():
                os.remove(f)

        # Khởi tạo LSH tương ứng
        t_init_start = time.perf_counter()
        
        if algo_choice == "Standard MinHash LSH":
            lsh = MinHashLSH(threshold=threshold, num_perm=num_perm, storage_config={"type": "dict"})
        elif algo_choice == "LSH Bloom":
            # Sử dụng tỷ lệ lỗi giả mặc định là 1e-5
            lsh = MinHashLSHBloom(threshold=threshold, num_perm=num_perm, fp=1e-5, n=len(lines), save_dir=str(temp_save_dir))
        else: # LSH BlowChoc
            lsh = MinHashLSHBlowChoc(threshold=threshold, num_perm=num_perm, fp=1e-5, n=len(lines), save_dir=str(temp_save_dir))
            
        t_init = (time.perf_counter() - t_init_start) * 1000
        
        results = []
        unique_hashes = {} # id -> minhash (dành cho MinHash LSH để đo độ tương đồng chính xác)
        unique_texts = {} # id -> text
        
        t_process_start = time.perf_counter()
        
        for idx, text in enumerate(lines):
            t_doc_start = time.perf_counter()
            
            # 1. Tạo chữ ký MinHash
            mh = build_minhash(text, num_perm, ngram_n)
            
            # 2. Truy vấn trùng lặp
            is_dup = False
            dup_reason = ""
            best_sim = 0.0
            
            if algo_choice == "Standard MinHash LSH":
                candidates = lsh.query(mh)
                # LSH truyền thống cần verify bằng Jaccard thực tế
                for cand_id in candidates:
                    sim = mh.jaccard(unique_hashes[cand_id])
                    if sim >= threshold:
                        is_dup = True
                        if sim > best_sim:
                            best_sim = sim
                            dup_reason = f"Trùng với Dòng {cand_id + 1} (Độ tương đồng Jaccard: {sim:.2%})"
            else:
                # LSH Bloom và LSH BlowChoc truy vấn trực tiếp qua bộ lọc Bloom rất nhanh
                is_dup = lsh.query(mh)
                if is_dup:
                    # Để hiển thị trực quan hơn, ta quét qua các câu duy nhất trước đó để chỉ ra câu trùng gần nhất
                    for cand_id, cand_mh in unique_hashes.items():
                        sim = mh.jaccard(cand_mh)
                        if sim >= threshold:
                            if sim > best_sim:
                                best_sim = sim
                                dup_reason = f"Trùng với Dòng {cand_id + 1} (Độ tương đồng Jaccard: {sim:.2%})"
                    if not dup_reason:
                        dup_reason = "Bị bộ lọc Bloom đánh dấu trùng lặp (Ứng viên tiềm năng)"
            
            t_doc = (time.perf_counter() - t_doc_start) * 1000
            
            # 3. Thêm vào LSH nếu duy nhất
            if not is_dup:
                if algo_choice == "Standard MinHash LSH":
                    lsh.insert(idx, mh)
                else:
                    lsh.insert(mh)
                unique_hashes[idx] = mh
                unique_texts[idx] = text
                
            results.append({
                "Dòng": idx + 1,
                "Nội dung văn bản": text,
                "Trạng thái": "🟢 Duy nhất (Unique)" if not is_dup else "🔴 Trùng lặp (Duplicate)",
                "Chi tiết phát hiện": dup_reason if is_dup else "Không trùng",
                "Thời gian xử lý (ms)": f"{t_doc:.3f} ms"
            })
            
        t_total = (time.perf_counter() - t_process_start) * 1000
        
        # Đồng bộ hóa bộ lọc lưu xuống đĩa nếu có
        if hasattr(lsh, "sync"):
            lsh.sync()
            
        # Hiển thị thống kê tổng quan
        st.subheader("📊 Thống kê hiệu năng khử trùng")
        col1, col2, col3 = st.columns(3)
        
        n_dups = sum(1 for r in results if "Trùng lặp" in r["Trạng thái"])
        
        col1.metric("Tổng số văn bản", len(lines))
        col2.metric("Số văn bản trùng lặp", n_dups, delta=f"-{(n_dups/len(lines)):.1%}" if len(lines) > 0 else "0%")
        col3.metric("Tổng thời gian xử lý", f"{t_total:.2f} ms")
        
        # Hiển thị bảng kết quả chi tiết
        st.subheader("📋 Bảng phân tích chi tiết từng dòng")
        st.dataframe(
            results,
            column_config={
                "Dòng": st.column_config.NumberColumn("STT", width=50),
                "Nội dung văn bản": st.column_config.TextColumn("Nội dung văn bản", width=500),
                "Trạng thái": st.column_config.TextColumn("Trạng thái", width=150),
                "Chi tiết phát hiện": st.column_config.TextColumn("Chi tiết phát hiện", width=300),
                "Thời gian xử lý (ms)": st.column_config.TextColumn("Thời gian xử lý", width=120),
            },
            hide_index=True,
            use_container_width=True
        )

# Dọn dẹp thư mục tạm khi đóng hoặc thoát (nếu có thể)
# Mặc dù streamlit chạy liên tục, thư mục tạm demo_temp nằm trong outputs/ là rất an toàn.
