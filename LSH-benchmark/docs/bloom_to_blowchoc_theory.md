# Lý thuyết: Từ Bloom Filter đến BlowChoc

> Chuỗi cải tiến từ cơ bản đến hiện đại, từng bước chỉ ra điểm yếu và giải pháp.

---

## Phần 1: Bài toán gốc — Dedup là gì?

Cho 1 triệu tài liệu, cần trả lời: **"Tài liệu này đã thấy trước đây chưa?"**

**Naive**: lưu tất cả tài liệu đã thấy vào một set → tốn RAM khổng lồ.

Cần một **cấu trúc xác suất**: chấp nhận một tỉ lệ nhỏ false positive để đổi lấy bộ nhớ nhỏ.

---

## Phần 2: Standard Bloom Filter

### Cấu trúc

```
m = 16 bits, k = 3 hash functions

Bit array:  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Index:       0 1 2 3 4 5 6 7 8 9 ...         15
```

### Insert("hello")

```
h1("hello") % 16 = 3
h2("hello") % 16 = 7
h3("hello") % 16 = 11

Bit array:  [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0]
                   ↑       ↑       ↑
```

### Query("hello") → True

```
Kiểm tra bit 3  → 1 ✓
Kiểm tra bit 7  → 1 ✓
Kiểm tra bit 11 → 1 ✓
→ Tất cả = 1: "Đã thấy" (True)
```

### Query("world") → False (Chắc chắn đúng)

```
h1("world") % 16 = 2  → bit 2 = 0 ✗
→ Ít nhất 1 bit = 0: "Chưa thấy" (False) — CHẮC CHẮN ĐÚNG
```

> **Nguyên lý cốt lõi**: Bloom Filter **không bao giờ false negative** (bỏ sót dup thật). Chỉ có thể false positive (báo dup nhầm).

### Công thức tối ưu

| Tham số | Công thức |
|---------|-----------|
| Số bits | $m = -\dfrac{n \ln \varepsilon}{(\ln 2)^2}$ |
| Số hash functions | $k = \lceil -\log_2 \varepsilon \rceil$ |
| FPR | $\varepsilon \approx (0.6185)^{m/n}$ |

---

## Phần 3: Vấn đề — Bloom thuần không đủ cho Dedup văn bản

Tài liệu là một đoạn văn dài → **không thể hash cả tài liệu** rồi so sánh bằng nhau
vì `"gần giống" ≠ "bằng nhau"`.

Cần đo **Jaccard Similarity**: doc A và doc B chia sẻ bao nhiêu % từ/cụm từ?

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

**Ví dụ:**
- Doc A = {"tôi", "đi", "học"}
- Doc B = {"tôi", "đi", "chơi"}
- $J(A, B) = 2/4 = 0.5$

Bloom Filter thuần không đo được Jaccard → cần **MinHash**.

---

## Phần 4: MinHash — ước lượng Jaccard

### Ý tưởng

Dùng $p$ hàm hash ngẫu nhiên, mỗi hàm "chọn phần tử nhỏ nhất" của tập từ.
Vector $p$ giá trị đó gọi là **MinHash signature**.

```
Doc A = {"tôi", "đi", "học"}
Signature A = [min_h1(A), min_h2(A), ..., min_h128(A)]
            = [3,          17,         ..., 82]
```

### Tính chất kỳ diệu

$$\Pr[h_{\min}(A) = h_{\min}(B)] = J(A, B)$$

Nếu Jaccard = 0.8 thì trong 128 hàm hash, trung bình **102 hàm** sẽ cho cùng giá trị.

### Vấn đề: so sánh pairwise tốn O(n²)

Với $n = 10^6$ docs, so sánh mọi cặp = $10^{12}$ phép so → không khả thi. Cần **LSH** để lọc nhanh.

---

## Phần 5: LSH (Locality Sensitive Hashing) — lọc nhanh

### Ý tưởng: chia bands

Chia 128 MinHash values thành $b$ bands, mỗi band có $r$ rows:

```
num_perm = 128, chia thành b=20 bands × r=6 rows

Band 0:  [mh[0],   mh[1],  ..., mh[5]]
Band 1:  [mh[6],   mh[7],  ..., mh[11]]
...
Band 19: [mh[114], ...,         mh[119]]
```

### Ngưỡng S-curve

Xác suất hai doc có Jaccard $s$ được phát hiện là dup:

$$P(\text{candidate}) = 1 - (1 - s^r)^b$$

```
b=20, r=6:

P(candidate)
1.0 |          ╭──────────
    |         /
0.5 |        /  ← ngưỡng T=0.5
    |       /
0.0 |──────╯
      0.0  0.5  1.0  → Jaccard
```

### Query logic: "OR of ANDs"

```python
for band in bands:
    if band_hash(query) ∈ stored_band_hashes:
        return True   # chỉ cần 1 band match → báo dup
return False
```

---

## Phần 6: LSH + Bloom Filter

### LSH thuần: lưu bands vào hashtable

```
Band 0: {"hash_band0_doc1": id1, "hash_band0_doc2": id2, ...}
Band 1: {...}
```

→ Tốn RAM vì phải lưu cả doc_id.

### LSH Bloom: thay hashtable bằng Bloom Filter

Với mỗi band, **không lưu doc_id** — chỉ hỏi: "Band hash này đã thấy chưa?"

```
Band 0:  [BloomFilter_0]
Band 1:  [BloomFilter_1]
...
Band b-1:[BloomFilter_b-1]
```

**Insert doc X:**
```
Với band i:
  key = hash(mh[i*r : (i+1)*r])  → 1 số nguyên
  BloomFilter_i.add(key)
```

**Query doc Y:**
```
Với band i:
  key = hash(mh[i*r : (i+1)*r])  → 1 số nguyên
  Nếu key ∈ BloomFilter_i → match → là dup
```

> **Đây chính xác là cách `lsh_bloom.py` vận hành** — dùng `MinHashLSHBloom` từ datasketch library.

### Vòng đời trong code (`lsh_bloom.py`)

```python
def deduplicate(self, text: str, id: int) -> bool:
    mh = self.get_minhash(text, id)    # tạo MinHash 128 values
    is_dup = self.lsh.query(mh)        # hỏi: band nào đã thấy chưa?
    if not is_dup:
        self.lsh.insert(mh)            # chưa thấy → add vào filter
    return is_dup
```

---

## Phần 7: ⚠️ Điểm yếu 1 của Standard Bloom — Cache thrashing

Giả sử $n = 10^6$ docs, $\varepsilon = 10^{-5}$:
- $m \approx 239$ MB
- $k = 17$ hash functions

**Query 1 band:**

```
h1(key) % 239MB → vị trí ngẫu nhiên  [cache miss #1]
h2(key) % 239MB → vị trí ngẫu nhiên  [cache miss #2]
...
h17(key)% 239MB → vị trí ngẫu nhiên  [cache miss #17]
```

**17 cache misses × 20 bands = 340 cache misses / document** → cực chậm.

> L3 cache miss ~40ns × 340 = **~13,600 ns** mỗi document.

---

## Phần 8: Blocked Bloom Filter (BBF) — giải pháp cache

### Ý tưởng

CPU load dữ liệu theo **cache line = 64 bytes = 512 bits**. Nhốt tất cả $k$ bits vào **1 cache line**:

```
Chia m bits thành các block 512 bits:

Block_0    Block_1    Block_2    ...    Block_N
[512 bits] [512 bits] [512 bits]       [512 bits]

h0(key) → chọn block B  (1 cache miss, load 64 bytes)
h1..hk(key) % 512 → k vị trí trong B  (không cache miss)
```

### Kết quả

| | Standard Bloom | Blocked Bloom |
|---|---|---|
| Cache misses/query | $k$ (~17) | **1** |
| Tốc độ | chậm | **~10× nhanh hơn** |
| FPR (cùng $m$) | $\varepsilon$ | ~1.74× tệ hơn |

### Tại sao FPR penalty 1.74×?

Vì $k$ bits bị constrain trong $B = 512$ bits thay vì tự do trong $m$ bits → các bits **ít độc lập hơn** → một số blocks trở nên "đầy" sớm → FPR của block đó tăng.

---

## Phần 9: ⚠️ Điểm yếu 2 của BBF — Sub-cache-line waste

Đã load 64 bytes (8 words × 8 bytes) vào cache. Nhưng $k=17$ bits rải **ngẫu nhiên** trong 512 bits:

```
Block = 8 words × 64 bits:

Word0[64] Word1[64] Word2[64] Word3[64] Word4[64] Word5[64] Word6[64] Word7[64]
  ↑↑                 ↑↑↑                  ↑↑↑↑                          ↑↑↑

17 bits của key rơi vào: Word0(2), Word2(3), Word4(4), Word7(3)
→ CPU đọc dữ liệu từ 4 words = 32 bytes thực dùng / 64 bytes load
→ Lãng phí 50% cache bandwidth
```

Xác suất tất cả $k=17$ bits nằm trong cùng 1 word:

$$(1/8)^{16} \approx 0\%$$

→ **Gần như chắc chắn phải đọc nhiều words** trong mỗi query.

---

## Phần 10: BlowChoc — nhốt tất cả vào 1 word

### Ý tưởng

> Thêm 1 lớp hash nữa: sau khi chọn block (cache line), chọn **đúng 1 word** trong block, rồi nhét tất cả $k$ bits vào word đó.

```
h0(key) → block_idx         [1 cache miss — load 64 bytes]
h1(key) → word_idx ∈ {0..7} [từ cache — chỉ dùng 8 bytes]
h2(key) % 64 → bit trong word
h3(key) % 64 → bit trong word
...
hk+1(key) % 64 → bit trong word
```

### Minh họa so sánh

```
Blocked Bloom — sau khi load block:
Word0  Word1  Word2  Word3  Word4  Word5  Word6  Word7
 ↑↑     ↑      ↑↑↑           ↑                   ↑↑↑
Đọc nhiều words → bandwidth lãng phí

BlowChoc — sau khi load block:
Word0  Word1  Word2  Word3  Word4  Word5  Word6  Word7
               ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
              Chỉ đọc Word2 — 8 bytes
```

### Tại sao FPR penalty chỉ ~5–10%?

Hai hiệu ứng **triệt tiêu nhau**:

| Hiệu ứng | Hướng |
|---------|-------|
| Word nhỏ hơn BBF 8× ($64$ vs $512$ bits) | FPR **tăng** |
| Mỗi word chỉ nhận $n/8$ inserts (chọn ngẫu nhiên trong 8 words) | FPR **giảm** |

Kết quả: penalty gần triệt tiêu, chỉ còn sai số nhỏ từ phân bố không hoàn toàn đều → **~5–10%**.

### Ràng buộc cứng: k ≤ 63

Nếu $k$ bits phải nằm trong 1 word 64-bit thì tối đa 64 vị trí (bit 0→63). Thực tế dùng 63. Giới hạn FPR tối thiểu:

$$\varepsilon_{\min} = 2^{-63} \approx 10^{-19}$$

### Implementation trong `lsh_blowchoc.py`

```python
def _get_word_and_bits(self, item: int):
    item_bytes = item.to_bytes(8, byteorder='little', signed=False)

    # Layer 1: chọn block (cache line)
    block_idx = mmh3.hash(item_bytes, seed=0) % self.num_blocks

    # Layer 2: chọn 1 word trong block
    word_idx = mmh3.hash(item_bytes, seed=1) % self.WORDS_PER_BLOCK

    # Layer 3: k bit positions trong word đó
    bit_positions = [
        mmh3.hash(item_bytes, seed=i+2) % self.WORD_BITS
        for i in range(self.k)
    ]
    return block_idx, word_idx, bit_positions

def add(self, item: int):
    block_idx, word_idx, bit_positions = self._get_word_and_bits(item)
    bit_pos_arr = np.array(bit_positions, dtype=np.uint64)
    masks = np.uint64(1) << bit_pos_arr
    combined_mask = np.bitwise_or.reduce(masks)       # 1 lần OR tất cả
    self.blocks[block_idx, word_idx] |= combined_mask # ghi vào 1 word
```

---

## Tổng kết chuỗi cải tiến

```
Standard Bloom Filter
  ✓ FPR tốt nhất (1×)
  ✗ k cache misses mỗi query (~17)
        │
        │ Vấn đề: quá nhiều cache miss → cực chậm
        ▼
Blocked Bloom Filter
  ✓ 1 cache miss mỗi query
  ✓ Load 64 bytes
  ✗ FPR penalty 1.74×
  ✗ Đọc nhiều words trong block → lãng phí bandwidth
        │
        │ Vấn đề: load 64 bytes nhưng thực dùng ít hơn nhiều
        ▼
BlowChoc
  ✓ 1 cache miss mỗi query
  ✓ Chỉ đọc 8 bytes (1 word) sau khi cache load
  ✓ FPR penalty nhỏ (~1.05–1.10×)
  ✗ k ≤ 63 (giới hạn cứng do word 64-bit)
        │
        │ Vấn đề: word được chọn ngẫu nhiên → load imbalance
        ▼
BlowChoc + Choices (Schmitz et al. 2025)
  ✓ Chọn word tốt nhất trong c candidates
  ✓ FPR giảm thêm ~20–30% nhờ load balancing
  ✗ Insert phức tạp hơn (cần tính cost function)
```

### Bảng so sánh

| | Standard Bloom | Blocked Bloom | BlowChoc | BlowChoc+Choices |
|---|:---:|:---:|:---:|:---:|
| Cache misses | $k$ (~17) | 1 | 1 | 1 |
| Bytes thực dùng | random | 64 | **8** | 8 |
| FPR penalty | 1× | 1.74× | **1.05–1.10×** | <1.05× |
| Tốc độ (tương đối) | 1× | ~10× | **~20×** | ~18× |
| Giới hạn $k$ | không | không | **≤ 63** | ≤ 63 |

---

## Tham khảo

- Putze, F., Sanders, P., Singler, J. (2010). *Cache-, Hash- and Space-Efficient Bloom Filters.*
- Breslow, A. D., Hutchings, N. S. (2018). *Morton Filters.*
- Schmitz, C. et al. (2025). *BlowChoc: Bloom Filter with Choices.*
- Leskovec, J., Rajaraman, A., Ullman, J. D. *Mining of Massive Datasets* — Chapter 3: Finding Similar Items.
