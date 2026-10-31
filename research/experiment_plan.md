# KẾ HOẠCH THỰC NGHIỆM VÀ THIẾT KẾ ĐỐI CHỨNG (EXPERIMENT PLAN)

## 1. Mục Tiêu Nghiên Cứu và Giả Thuyết Khoa Học

### 1.1. Mục tiêu
Nghiên cứu, triển khai và đánh giá khách quan phương pháp **NBAC-PhoBERT (Neutral-Boundary-Aware Contrastive Fine-tuning)** nhằm giải quyết bài toán phân loại cảm xúc trên corpus UIT-VSFC, đặc biệt khắc phục điểm nghẽn hiệu năng trên lớp thiểu số **Neutral** (chỉ chiếm ~4,01% tập train).

### 1.2. Giả thuyết nghiên cứu
1. **H1 (Hiệu quả của Neutral Boundary Loss)**: Khi kết hợp với Cross-Entropy, hàm loss định hướng ranh giới Neutral $\mathcal{L}_{\text{NB}}$ giúp phân tách tốt hơn giữa biểu diễn các câu Neutral và các câu Negative/Positive cận ranh giới, giảm thiểu việc mô hình nhầm lẫn các câu trung tính thành phân cực.
2. **H2 (Đóng góp độc lập của Loss so với Sampler)**: Sự cải thiện (nếu có) của NBAC thực sự đến từ thành phần hình học của $\mathcal{L}_{\text{NB}}$, chứ không đơn thuần do việc class-aware sampler cung cấp nhiều mẫu Neutral hơn trong mỗi batch. (Kiểm chứng qua so sánh B4 vs B6).
3. **H3 (Ưu thế so với Contrastive đối xứng)**: $\mathcal{L}_{\text{NB}}$ tập trung vào hard negatives của riêng lớp Neutral sẽ hiệu quả hơn việc kéo thả toàn bộ các lớp một cách đồng đều như Supervised Contrastive Loss tiêu chuẩn (SupCon). (Kiểm chứng qua so sánh B5 vs B6).
4. **H4 (Hiệu chỉnh logit/bias hậu nghiệm)**: Việc tinh chỉnh bias Neutral $b_N$ trên tập Validation cho phép điều hướng trade-off giữa Precision và Recall một cách có kiểm soát mà không làm biến dạng biểu diễn đặc trưng (B7 vs B6).

---

## 2. Hệ Thống Cấu Hình Thí Nghiệm (Baselines và Ablations)

Mọi cấu hình deep learning đều cố định:
- **Backbone**: `vinai/phobert-base` (135M tham số) tải qua safetensors cục bộ.
- **Preprocessing**: Tách từ bằng PyVi (`ViTokenizer.tokenize`), BPE tokenization với PhoBERT tokenizer, `max_length = 96`.
- **Optimizer**: AdamW, learning rate = $2\times 10^{-5}$, weight decay = $0.01$.
- **Batch size**: 32, FP16 mixed precision, training 5 epochs với early stopping và model checkpointing dựa trên Validation Macro-F1.

| Mã | Tên cấu hình | Backbone | Sampler | Loss Function | Post-hoc Calibration | Mục đích đối chứng |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B0a** | TF-IDF + LR (None) | N/A | Random | Logistic Loss (None) | Không | Baseline tuyến tính truyền thống (tái lập số liệu bản nháp) |
| **B0b** | TF-IDF + LR (Balanced) | N/A | Random | Logistic Loss (Balanced) | Không | Minh họa trade-off Precision-Recall khi dùng cost-sensitive |
| **B1** | PhoBERT + CE | PhoBERT-base | Standard Random | Standard Cross-Entropy | Không | Baseline Deep Learning chuẩn mực |
| **B2** | PhoBERT + Weighted CE | PhoBERT-base | Standard Random | Class-Weighted Cross-Entropy | Không | So sánh với phương pháp re-weighting cổ điển |
| **B3** | PhoBERT + CE + Val Bias | PhoBERT-base | Standard Random | Standard Cross-Entropy | Tối ưu $b_N$ trên Validation | Đánh giá hiệu quả của riêng việc dịch chuyển logit |
| **B4** | PhoBERT + CE + Sampler | PhoBERT-base | Class-Aware Sampler | Standard Cross-Entropy | Không | **Ablation cốt lõi**: Đo lường tác động của riêng Sampler |
| **B5** | B4 + Standard SupCon | PhoBERT-base | Class-Aware Sampler | $\mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{SupCon}}$ | Không | Đối chứng với Contrastive Learning đối xứng đa lớp |
| **B6** | B4 + NBAC Loss | PhoBERT-base | Class-Aware Sampler | $\mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{NB}}$ | Không | **Đề xuất chính**: Đánh giá Neutral Boundary Loss |
| **B7** | B6 + Val Bias | PhoBERT-base | Class-Aware Sampler | $\mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{NB}}$ | Tối ưu $b_N$ trên Validation | Hệ thống hoàn chỉnh đề xuất |

---

## 3. Chi Tiết Toán Học và Xử Lý Biên

### 3.1. Neutral Boundary Loss ($\mathcal{L}_{\text{NB}}$)
Cho batch kích thước $B$, biểu diễn đặc trưng chuẩn hóa $h_i = \frac{z_i}{\|z_i\|_2} \in \mathbb{R}^d$ lấy từ pooled representation (hoặc [CLS] token) của PhoBERT.
Hàm độ tương đồng cosine:

$$
s(a, b) = h_a^\top h_b
$$

Ký hiệu $\mathcal{N} = \{i \mid y_i = \text{Neutral}\}$ là tập các chỉ số mẫu Neutral trong batch.
Với mỗi anchor $i \in \mathcal{N}$:
1. **Positive $p_i$**: Mẫu Neutral khác trong batch. Để đảm bảo tính chặt chẽ, ta chọn hardest positive trong batch:

   $$
   p_i = \operatorname*{arg\,min}_{j \in \mathcal{N}, j \neq i} s(h_i, h_j)
   $$

   (hoặc trung bình trên tất cả các mẫu positive có mặt trong batch).
2. **Hard Negative $q_i$**: Mẫu thuộc lớp Negative hoặc Positive có độ tương đồng cosine lớn nhất với $h_i$:

   $$
   q_i = \operatorname*{arg\,max}_{k \notin \mathcal{N}} s(h_i, h_k)
   $$

3. **Hàm mất mát biên**:

   $$
   \mathcal{L}_{\text{NB}} = \frac{1}{|\mathcal{N}_{\text{valid}}|} \sum_{i \in \mathcal{N}_{\text{valid}}} \max\left(0, m - s(h_i, h_{p_i}) + s(h_i, h_{q_i})\right)
   $$

### 3.2. Xử lý các Edge Cases trong PyTorch
- **Trường hợp $|\mathcal{N}| = 0$ (Không có mẫu Neutral trong batch)**: $\mathcal{L}_{\text{NB}} = 0.0 \times \sum h$. Không phát sinh lỗi chia cho 0, không sinh NaN, bảo toàn đồ thị gradient.
- **Trường hợp $|\mathcal{N}| = 1$ (Chỉ có 1 mẫu Neutral trong batch)**: Không thể tạo cặp positive $(h_i, h_p)$ hợp lệ → loại anchor này khỏi tính toán loss. Nếu không còn anchor hợp lệ, trả về tensor 0 có liên kết gradient.
- **Trường hợp batch toàn bộ là Neutral (Không có Negative hoặc Positive)**: Không có negative $q$ → bỏ qua.
- **Grad clip & Numerical stability**: Sử dụng `clamp` khi tính khoảng cách và chuẩn hóa vector để tránh tràn số hoặc chia cho 0.

### 3.3. Thiết Kế Class-Aware Batch Sampler
Nhằm khắc phục tình trạng batch ngẫu nhiên thiếu vắng mẫu Neutral:
- Mỗi batch kích thước $B = 32$ được lấy mẫu có chủ đích sao cho luôn có ít nhất $k_{neu} = 6$ mẫu Neutral, và phần còn lại phân bổ cân đối giữa Negative ($13$ mẫu) và Positive ($13$ mẫu).
- Thứ tự mẫu trong từng lớp được xáo trộn ngẫu nhiên mỗi epoch. Mẫu lớp thiểu số được quay vòng (oversampled with replacement) trong khi vẫn đảm bảo duyệt qua toàn bộ dữ liệu lớp đa số.

---

## 4. Giao Thức Chống Data Leakage và Nguyên Tắc Lựa Chọn

1. **Tuyệt đối không dùng Test để điều chỉnh tham số**:
   - Tất cả các quyết định: chọn checkpoint tốt nhất theo epoch, điều chỉnh margin $m$, hệ số $\lambda$, tìm bias $b_N$ đều **100% thực hiện trên tập Validation**.
   - Tập Test chỉ được nạp và đánh giá **đúng 1 lần duy nhất** sau khi đã chốt mô hình và tham số.
2. **Không gian tìm kiếm tham số (Hyperparameter Grid)**:
   - $\lambda \in \{0.05, 0.1, 0.2\}$
   - $m \in \{0.1, 0.2, 0.3\}$
   - Neutral bias $b_N \in [-1.5, 1.5]$ với bước nhảy $\Delta = 0.05$ (được quét trên logit Validation sau khi mô hình đã hội tụ).
3. **Tính trung thực về Selection Bias**:
   - Ghi rõ trong báo cáo rằng kết quả baseline trong bản nháp đã được biết từ trước khi lập kế hoạch, do đó các giả thuyết được hình thành có thể chịu ảnh hưởng tiềm tàng của selection bias.

---

## 5. Metrics Đánh Giá và Phương Pháp Thống Kê

1. **Metric chính**: **Macro-F1** (trung bình không trọng số của F1 cả 3 lớp).
2. **Metrics lớp Neutral**:
   - Neutral Precision, Neutral Recall, Neutral F1.
   - Số lượng False Positives (FP) và False Negatives (FN) của Neutral.
3. **Metrics bổ trợ**:
   - Per-class F1 (Negative, Positive), Overall Accuracy, Weighted-F1.
   - Confusion Matrix $3 \times 3$.
4. **Phân tích độ bất định và kiểm định thống kê**:
   - Huấn luyện đa seed (ví dụ: seeds `42, 100, 2024, 777, 999`) cho các cấu hình chính để tính Mean ± Standard Deviation.
   - Phân tích Paired Bootstrap (1.000 lần lấy mẫu lại có hoàn lại trên tập Test) để tính Khoảng tin cậy 95% (95% Confidence Interval) cho Macro-F1 và Neutral F1.
   - Kiểm định McNemar / Paired t-test để so sánh mức độ ý nghĩa thống kê ($p$-value $< 0.05$) giữa mô hình đề xuất và baseline B1, B2.

---

## 6. Tiêu Chí Đánh Giá Thành Bại (Acceptance Criteria)

Phương pháp NBAC-PhoBERT chỉ được kết luận là thành công nếu thỏa mãn đồng thời:
1. Macro-F1 trên tập Test cao hơn B1 (PhoBERT + CE tiêu chuẩn) một cách có ý nghĩa thống kê ($p < 0.05$).
2. Neutral F1 cải thiện rõ rệt, không đánh đổi bằng việc sụp đổ Precision như B2 (Weighted CE).
3. Hiệu năng trên Negative và Positive được bảo toàn (chênh lệch F1 không quá 1%).
4. Ablation chứng minh $\mathcal{L}_{\text{NB}}$ (B6) vượt trội hơn cả việc chỉ dùng sampler (B4) và dùng SupCon tiêu chuẩn (B5).
5. Nếu kết quả thực tế không vượt trội, **báo cáo trung thực kết quả âm**, phân tích chi tiết nguyên nhân ngữ nghĩa học hoặc hình học biểu diễn.
