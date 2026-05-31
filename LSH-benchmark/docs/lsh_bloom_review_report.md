# Báo Cáo Đánh Giá Kỹ Thuật: Phương Pháp LSH Bloom (Locality Sensitive Hashing with Bloom Filters)

---

## 1. Tổng Quan Phương Pháp

Phương pháp **LSH Bloom** (được triển khai qua lớp `MinHashLSHBloom` trong thư viện datasketch) là một bước tiến quan trọng trong bài toán khử trùng lặp dữ liệu văn bản quy mô lớn. 

Thay vì sử dụng các cấu trúc bảng băm (Hash Table) truyền thống để lưu trữ danh sách chữ ký MinHash và ID tài liệu, LSH Bloom tích hợp **Bộ lọc xác suất Bloom (Bloom Filter)** làm tầng lưu trữ chỉ mục. Phương pháp này giải quyết trực tiếp bài toán quá tải bộ nhớ RAM khi số lượng tài liệu cần lọc tăng lên hàng triệu hoặc hàng tỷ bản ghi.

---

## 2. Nguyên Lý Vận Hành Chi Tiết

Quy trình hoạt động của LSH Bloom dựa trên sự kết hợp giữa kỹ thuật chia nhóm LSH và tính toán xác suất của bộ lọc Bloom:

1.  **Tính chữ ký MinHash:** Mỗi văn bản đầu vào được chuyển đổi thành một danh sách chữ ký số (Signature) gồm $k$ phần tử.
2.  **Chia Band (Banding):** Chữ ký MinHash được chia thành $b$ bands riêng biệt, mỗi band chứa $r$ hàng ($b \times r = k$).
3.  **Ánh xạ sang bộ chỉ mục Bloom:** Hệ thống khởi tạo $b$ bộ lọc Bloom độc lập, mỗi bộ lọc đại diện cho một band.
4.  **Băm và Kiểm tra:** 
    *   Đối với mỗi band $i$ của tài liệu, hệ thống lấy $r$ giá trị số nguyên, băm chúng lại thành một mã số nhận diện duy nhất.
    *   Mã số này được truy vấn vào bộ lọc Bloom thứ $i$. Nếu mã băm này tồn tại trong bộ lọc Bloom của **ít nhất một band**, tài liệu lập tức bị đánh dấu là trùng lặp.
    *   Nếu chưa tồn tại trong tất cả các band, mã băm sẽ được chèn vào các bộ lọc Bloom tương ứng để lưu vết cho các tài liệu tiếp theo.

---

## 3. Các Ưu Điểm Nổi Bật

### 3.1. Dung lượng bộ nhớ RAM đứng im cố định (Static Memory Allocation)
Đây là thế mạnh tuyệt đối của LSH Bloom so với LSH truyền thống. 
*   **Với LSH truyền thống:** Bộ nhớ phình to tuyến tính theo số lượng tài liệu vì phải lưu trữ các chuỗi ID tài liệu và chữ ký băm.
*   **Với LSH Bloom:** Bộ nhớ được cấp phát tĩnh ngay từ khi khởi động dựa trên số lượng phần tử dự kiến và tỷ lệ báo sai (FPR) mong muốn. Việc thêm tài liệu mới chỉ là thao tác bật các bit từ `0` thành `1` trên mảng bộ nhớ có sẵn, hoàn toàn không sinh ra ô nhớ mới. RAM giữ nguyên một đường thẳng tắp trong suốt quá trình khử trùng, triệt tiêu nguy cơ sập hệ thống do lỗi tràn bộ nhớ (Out-Of-Memory).

### 3.2. Không rò rỉ thông tin gốc (Data Privacy)
Bộ lọc Bloom chỉ lưu trữ trạng thái tồn tại của các mã băm dạng bit, hoàn toàn không lưu trữ ID tài liệu hay nội dung gốc của chữ ký. Điều này đảm bảo tính bảo mật dữ liệu tuyệt đối, đặc biệt hữu ích trong các hệ thống xử lý thông tin nhạy cảm.

---

## 4. Điểm Yếu Chí Tử & Nút Thắt Cổ Chai Phần Cứng

Dù tối ưu vượt trội về mặt dung lượng RAM, LSH Bloom truyền thống gặp phải các vấn đề nghiêm trọng về hiệu năng xử lý phần cứng:

### 4.1. Hiện tượng nghẽn bộ đệm CPU (L3 Cache Thrashing)
*   **Bản chất:** Bộ lọc Bloom của LSH Bloom được phân bổ ngẫu nhiên trên toàn bộ không gian RAM khổng lồ (hàng trăm MB). 
*   **Nút thắt:** Mỗi lần kiểm tra hoặc chèn một mã băm, CPU phải truy xuất RAM ở $k$ (thường từ 10 đến 20) địa chỉ ngẫu nhiên nằm cách xa nhau.
*   **Hậu quả:** CPU liên tục bị trượt bộ đệm (**L3 Cache Misses**). Ở quy mô lớn, việc CPU phải dừng lại hàng trăm lần để đợi RAM nạp dữ liệu cho mỗi tài liệu khiến tốc độ xử lý toàn hệ thống bị kéo chậm đi nghiêm trọng.

### 4.2. Lãng phí băng thông bus dữ liệu nội bộ
Mỗi lần CPU muốn đọc 1 bit ngẫu nhiên từ RAM, phần cứng bắt buộc phải nạp nguyên 1 khối 64-byte (Cache Line) chứa bit đó vào cache. Với LSH Bloom, $k$ bit nằm ở $k$ khối cách xa nhau, nghĩa là CPU nạp rất nhiều khối dữ liệu vào cache nhưng chỉ sử dụng đúng 1 bit duy nhất trong mỗi khối $\rightarrow$ Gây lãng phí cực lớn băng thông bus dữ liệu của hệ thống.

---

## 5. Hướng Cải Tiến: Cầu Nối Đến WA-BBF

Chính những điểm yếu chí tử về mặt phần cứng của LSH Bloom đã mở đường cho sự ra đời của **WA-BBF (Word-Aligned Blocked Bloom Filter)**:

1.  **Gom cụm dữ liệu (Blocked):** WA-BBF sửa lỗi Cache Thrashing bằng cách nhốt toàn bộ bits băm vào chung một khối 64-byte duy nhất (vừa khít 1 Cache Line). CPU chỉ tốn đúng 1 lần đọc RAM cho mỗi truy vấn.
2.  **Căn lề từ nhớ (Word-Aligned):** WA-BBF sửa lỗi lãng phí bus bằng cách ép toàn bộ bits băm của phần tử vào đúng 1 từ nhớ 8-byte (uint64) trong khối. CPU chỉ thao tác đúng 1 lệnh bitwise trên thanh ghi, tăng tốc xử lý lên **gấp 8 lần** so với LSH Bloom truyền thống trong khi vẫn giữ nguyên đặc tính RAM cố định tuyệt vời.
