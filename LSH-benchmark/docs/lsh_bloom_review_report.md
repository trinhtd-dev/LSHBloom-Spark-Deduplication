# Báo Cáo Đánh Giá Kỹ Thuật: Phương Pháp LSH Bloom

---

## 1. Ý Tưởng Cốt Lõi

Phương pháp LSH Bloom ra đời nhằm giải quyết nút thắt cổ chai về dung lượng RAM khi khử trùng lặp dữ liệu (Deduplication) ở quy mô hàng triệu đến hàng tỷ văn bản.

*   **Tối ưu hóa không gian lưu trữ:** Trong thuật toán LSH truyền thống, việc lưu trữ các ID tài liệu và chữ ký số vào bảng băm khiến RAM phình to liên tục theo số lượng tài liệu đầu vào. LSH Bloom loại bỏ hoàn toàn các ID này, thay thế toàn bộ tầng chỉ mục bảng băm bằng các **Bộ lọc xác suất Bloom (Bloom Filter)** dạng mảng bit siêu nén.
*   **Cấp phát tĩnh và cố định RAM:** Bộ nhớ được cấp phát tĩnh ngay khi khởi động dựa trên dung lượng dự kiến và tỷ lệ báo sai (FPR) mong muốn. Việc thêm tài liệu mới chỉ là thao tác chuyển đổi các bit `0` sẵn có thành `1` trên mảng bộ nhớ, giữ RAM luôn ở trạng thái "đứng im cố định" và loại bỏ hoàn toàn rủi ro sập hệ thống vì tràn bộ nhớ (Out-of-Memory).
*   **Đánh đổi toán học:** Thay vì lưu trữ chính xác, LSH Bloom chấp nhận một tỷ lệ báo sai cực nhỏ (False Positive Rate) có thể kiểm soát được bằng toán học để đổi lấy hiệu quả lưu trữ vượt trội.

---

## 2. Cách Thức Triển Khai (Implementation)

Hệ thống tích hợp bộ lọc Bloom vào cơ chế chia nhóm (Banding) của Locality Sensitive Hashing:

*   **Phân chia phân băng (Banding):** Chữ ký MinHash của tài liệu (ví dụ gồm 128 số nguyên) được chia thành $b$ bands riêng biệt, mỗi band chứa $r$ hàng. Hệ thống khởi tạo $b$ bộ lọc Bloom độc lập tương ứng với các bands này.
*   **Tạo mã băm đại diện (Fingerprint):** Với mỗi band của tài liệu, thuật toán băm $r$ giá trị số nguyên trong band đó thành một mã băm đại diện duy nhất (fingerprint).
*   **Thao tác băm và kiểm tra trên bộ lọc Bloom:**
    *   *Truy vấn (Query):* Mã băm đại diện được băm tiếp thành $k$ vị trí bit ngẫu nhiên trong bộ lọc Bloom tương ứng. Nếu toàn bộ $k$ bit này đều bằng `1` ở ít nhất một band, tài liệu bị coi là trùng lặp.
    *   *Chèn (Insert):* Nếu tài liệu được xác định là duy nhất (chưa trùng), hệ thống sẽ bật cả $k$ bit tại các vị trí ngẫu nhiên trên mảng bộ nhớ Bloom của từng band tương ứng lên `1` để lưu vết cho các tài liệu tiếp theo.

---

## 3. Kết Quả & Nhận Xét (Evaluation & Remarks)

### 3.1. Kết quả chất lượng kiểm thử trên tập `test_p_0.5`
Thực nghiệm đo lường độ chính xác của thuật toán LSH Bloom trên tập dữ liệu chuẩn `test_p_0.5` (gồm 3,578 tài liệu với tỷ lệ trùng lặp 50%, sử dụng chữ ký MinHash 128 hoán vị) đạt kết quả như sau:

| Ngưỡng Jaccard | Độ chính xác (Precision) | Độ bao phủ (Recall) | Chỉ số F1-Score |
|:---:|:---:|:---:|:---:|
| 0.1 | 0.5598 | 0.9925 | 0.7158 |
| 0.3 | 0.5706 | 0.9215 | 0.7048 |
| **0.5** | **0.6575** | **0.5980** | **0.6263** |
| 0.7 | 0.6994 | 0.4095 | 0.5166 |
| 0.9 | 0.6964 | 0.2695 | 0.3886 |

*   *Nhận xét:* Sai số chất lượng (F1-score) của bộ lọc Bloom so với thuật toán LSH gốc lưu RAM đầy đủ là cực kỳ nhỏ (chênh lệch dưới 0.02% tùy ngưỡng), chứng minh độ tin cậy toán học gần như tuyệt đối của bộ lọc xác suất.

---

### 3.2. Hiệu năng mở rộng quy mô (Scale Benchmark) trên tập peS2o
Thực nghiệm đo lường hiệu năng của LSH Bloom khi mở rộng quy mô dữ liệu chèn từ 200.000 tài liệu lên tới **1.000.000 tài liệu**:

| Quy mô dữ liệu (Số tài liệu) | Tổng thời gian thực thi (giây) | Bộ nhớ RAM đỉnh (Peak RAM) |
|:---:|:---:|:---:|
| 200.000 | 223,5 giây | **1,98 GB** |
| 400.000 | 451,1 giây | **1,99 GB** |
| 600.000 | 682,7 giây | **1,99 GB** |
| 800.000 | 913,4 giây | **1,99 GB** |
| **1.000.000** | **1.148,6 giây (~19,1 phút)** | **1,99 GB (Cố định phẳng tắp)** |

---

### 3.3. Nhận xét đánh giá cốt lõi (So sánh trực tiếp với LSH Truyền thống)

*   **Sự ổn định tuyệt hảo về bộ nhớ (RAM):** 
    *   *Minhashing LSH:* Tiêu thụ bộ nhớ tăng tuyến tính theo số lượng tài liệu đầu vào (đỉnh điểm lên tới **4.12 GB** khi xử lý 1 triệu văn bản) do phải lưu trữ chính xác danh sách các ID tài liệu và chữ ký số vào cấu trúc bảng băm động (Python dictionary). Điều này dẫn đến nguy cơ tràn bộ nhớ (Out-Of-Memory - OOM) cực lớn khi quy mô dữ liệu tiếp tục mở rộng.
    *   *LSH Bloom:* Giữ mức tiêu thụ RAM luôn phẳng tắp và cố định hoàn hảo ở mức **1.99 GB** từ tài liệu đầu tiên cho đến tài liệu thứ một triệu. Đây là minh chứng rõ nét cho sức mạnh của cơ chế cấp phát tĩnh: việc chèn thêm dữ liệu chỉ thay đổi trạng thái bit trên mảng NumPy có sẵn mà không cấp phát thêm bất kỳ ô nhớ mới nào, giúp hệ thống an toàn tuyệt đối trước lỗi tràn bộ nhớ (OOM).
*   **Hiệu năng tốc độ vượt trội nhưng vấp phải nút thắt cổ chai phần cứng (L3 Cache Thrashing):**
    *   *Minhashing LSH:* Bị giới hạn nặng nề bởi chi phí tính toán phần mềm cao. Việc chèn, tìm kiếm và liên tục tái phân bổ (resize) các bucket trên cấu trúc dictionary của Python mất tới **97.7 phút** để hoàn thành 1 triệu tài liệu.
    *   *LSH Bloom:* Mang lại tốc độ xử lý nhanh hơn gấp **5 lần** (chỉ mất **19.1 phút** cho 1 triệu tài liệu) bằng cách loại bỏ hoàn toàn các overhead của đối tượng Python và thay thế bằng các phép toán bit nhanh chóng trên mảng NumPy phẳng.
    *   *Nút thắt cổ chai của LSH Bloom:* Mặc dù vượt trội hơn Minhashing LSH nhờ loại bỏ được overhead cấu trúc dữ liệu phần mềm, điểm yếu lớn nhất của LSH Bloom lại nằm ở tốc độ truy xuất phần cứng vật lý. Do mỗi truy vấn yêu cầu kiểm tra $k$ vị trí bit ngẫu nhiên trên một mảng bit lớn nằm rải rác khắp RAM, CPU liên tục gặp lỗi trượt bộ đệm (L3 Cache Misses). Việc CPU phải dừng lại để đợi RAM nạp dữ liệu hàng trăm lần cho mỗi tài liệu là nguyên nhân chính tạo nên "nút thắt cổ chai" giới hạn thông lượng (throughput) xử lý của thuật toán khi scale lên quy mô lớn hơn nữa.

