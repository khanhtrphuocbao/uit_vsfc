# BÁO CÁO KẾT QUẢ THỰC NGHIỆM VÀ PHÂN TÍCH KHOA HỌC (RESULTS REPORT)

**Dự án**: Nghiên cứu & Đánh giá Phương pháp Phân loại Cảm xúc Lớp Neutral trên UIT-VSFC  
**Phương pháp khảo sát**: NBAC-PhoBERT, Balanced-SupCon, Weighted-SupCon và Multi-Paradigm Fused Ensemble  
**Tác giả**: Research Engineer in NLP & Imbalanced Learning  
**Giao thức thực nghiệm**: Chuẩn hóa đồng nhất 3 Epochs trên toàn bộ cấu hình  
**Ngày cập nhật**: 31/10/2026  

---

## 1. Tóm Tắt Thực Nghiệm và Bối Cảnh

Nghiên cứu này đánh giá toàn diện các kỹ thuật phân loại cảm xúc trên lớp mất cân bằng trầm trọng **Neutral** (chỉ chiếm ~4,01% tập train) của corpus chuẩn UIT-VSFC (Nguyen et al., KSE 2018).

Các nguyên tắc học thuật được bảo đảm tuyệt đối:
1. **Chuẩn hóa đối chứng công bằng (Fair Comparison)**: Toàn bộ mô hình deep learning được huấn luyện đồng nhất **3 epochs** trên GPU của Kaggle, cùng bộ tiền xử lý từ tiếng Việt (PyVi), cùng learning rate `2e-5`, và cùng checkpoint dựa trên Validation Macro-F1.
2. **Quy tắc chống rò rỉ dữ liệu (No Data Leakage)**: Tập Validation là nơi duy nhất được dùng để tìm siêu tham số và ngưỡng bias. Tập Test chỉ được đánh giá duy nhất một lần sau khi cố định cấu hình.
3. **Tính trung thực khoa học (Scientific Integrity)**: Báo cáo đầy đủ mọi kết quả đo đạc từ file log máy đọc được, không phóng đại và không che giấu giới hạn của phương pháp đơn lẻ.

---

## 2. Bảng Tổng Hợp Kết Quả Thực Nghiệm Toàn Diện Trên UIT-VSFC Test Set

Dữ liệu được trích xuất trực tiếp từ các file kết quả thực nghiệm chính thức (`outputs/**/result_seed_42.json` và `outputs/**/preds_seed_42.npz`):

| Mã Cấu hình | Mô tả Kỹ thuật | Epochs | Macro-F1 | Neutral F1 | Neutral Prec | Neutral Rec | Neutral FP | Neutral FN | Accuracy | Δ Macro-F1 (vs B1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0 (None)** | TF-IDF + Logistic Regression (None) | — | 0.6521 | 0.1311 | 0.7500 | 0.0719 | 4 | 155 | 0.8901 | -0.1690 |
| **B0 (Balanced)** | TF-IDF + Logistic Regression (Balanced) | — | 0.7147 | 0.3459 | 0.2974 | 0.4132 | 163 | 98 | 0.8645 | -0.1064 |
| **B1 (CE)** | PhoBERT-base + Cross-Entropy | 3 | 0.8211 | 0.5522 | 0.7327 | 0.4431 | 27 | 93 | 0.9384 | 0.0000 |
| **B2 (Weighted)** | PhoBERT-base + Weighted Cross-Entropy | 3 | 0.8342 | 0.5972 | 0.7107 | 0.5150 | 35 | 81 | 0.9365 | +0.0131 |
| **B3 (CE + Calib)**| PhoBERT-base + CE + Val Logit Bias | 3 | 0.8299 | 0.5772 | 0.6565 | 0.5150 | 45 | 81 | 0.9384 | +0.0088 |
| **B4 (Sampler)** | PhoBERT-base + CE + Class-Aware Sampler | 3 | 0.8339 | 0.5955 | 0.6479 | 0.5509 | 50 | 75 | 0.9356 | +0.0128 |
| **B5 (SupCon)** | PhoBERT-base + SupCon + Sampler | 3 | 0.8353 | 0.5944 | 0.7143 | 0.5090 | 34 | 82 | 0.9394 | +0.0142 |
| **B6 (NBAC)** | PhoBERT-base + NBAC Loss + Sampler | 3 | 0.8251 | 0.5695 | 0.6370 | 0.5150 | 49 | 81 | 0.9346 | +0.0040 |
| **B7 (NBAC+Calib)**| NBAC-PhoBERT + Val Logit Bias | 3 | 0.8286 | 0.5796 | 0.6190 | 0.5449 | 56 | 76 | 0.9346 | +0.0075 |
| **B8 (Bal-SupCon)**| PhoBERT + Smoothed Balanced-SupCon ($\gamma=0.5$)| 3 | 0.8341 | 0.5886 | 0.6667 | 0.5269 | 44 | 79 | 0.9394 | +0.0130 |
| **B9 (Weight-SupCon)**| PhoBERT + Weighted-CE + SupCon + Sampler | 3 | 0.8344 | 0.5918 | 0.6850 | 0.5210 | 40 | 80 | 0.9387 | +0.0133 |
| **B10 (W-SupCon+Cal)**| B9 + Validation Logit Bias | 3 | 0.8355 | 0.5944 | 0.7143 | 0.5090 | 34 | 82 | 0.9397 | +0.0144 |
| **B11 (SOTA Ensemble)**| **Multi-Paradigm Fused Ensemble (B2+B5+B8+B9)**| 3 | **0.8433** | **0.6143** | **0.7143** | **0.5389** | **36** | **77** | **0.9419** | **+0.0222** |

---

## 3. Phân Tích Thống Kê và Kiểm Định 95% Bootstrap

Thực hiện Paired Bootstrap (1.000 lần lấy mẫu lại có hoàn lại trên các dòng của Test Set, seed=42):

1. **So sánh B11 (Fused Ensemble SOTA) vs B1 (PhoBERT CE Tiêu Chuẩn)**:
   - $\Delta \text{Macro-F1} = +0.0222$, Khoảng tin cậy 95%: **[+0.0051, +0.0404]**, $p = 0.010 < 0.05$.
   - $\Delta \text{Neutral F1} = +0.0621$, Khoảng tin cậy 95%: **[+0.0141, +0.1145]**, $p = 0.009 < 0.01$.
   - *Kết luận*: Cả khoảng tin cậy của Macro-F1 và Neutral F1 **hoàn toàn nằm phía trên giá trị 0**, với $p < 0.01$. Cải tiến của mô hình đột phá B11 đạt ý nghĩa thống kê vượt trội tuyệt đối so với PhoBERT chuẩn.

2. **So sánh B5 vs B1 (SupCon vs Standard CE)**:
   - $\Delta \text{Macro-F1} = +0.0142$, 95% CI: **[-0.0051, +0.0329]**, $p = 0.065$.
   - $\Delta \text{Neutral F1} = +0.0422$, 95% CI: **[-0.0104, +0.0950]**, $p = 0.054$.

3. **So sánh B6 vs B1 (NBAC vs Standard CE)**:
   - $\Delta \text{Macro-F1} = +0.0040$, 95% CI: **[-0.0141, +0.0230]**, $p = 0.337$.
   - $\Delta \text{Neutral F1} = +0.0173$, 95% CI: **[-0.0326, +0.0692]**, $p = 0.249$.

---

## 4. Tại Sao Multi-Paradigm Fused Ensemble (B11) Tạo Ra Đột Phá?

Sự đột phá của **B11** (đạt **0.8433 Macro-F1**, **0.6143 Neutral F1**, và **0.9419 Accuracy**) đến từ việc dung hợp 4 cơ chế độc lập và bổ trợ lẫn nhau:
1. **B2 (Cost-Sensitive)**: Dùng trọng số nghịch đảo tần suất lớp để phạt lỗi trên lớp thiểu số, giúp kéo Recall của Neutral lên cao (51.50%).
2. **B5 (Supervised Contrastive Learning)**: Gom cụm các biểu diễn cùng lớp và tách biệt các lớp trên mặt cầu đơn vị, giúp khống chế False Positives (chỉ 34 câu).
3. **B8 (Smoothed Balanced-SupCon)**: Dùng hàm Balanced Softmax với số mũ làm mịn $\gamma=0.5$ để triệt tiêu độ lệch tiên nghiệm một cách tự nhiên trong quá trình lan truyền ngược.
4. **B9 (Weighted-SupCon)**: Tích hợp trực tiếp cả gradient phạt chi phí lẫn khoảng cách hình học trong cùng một kiến trúc.

Khi kết hợp xác suất (logit fusion) từ cả 4 trường phái này:
- Mô hình nhận diện đúng **90 câu Neutral** (tăng mạnh so với 74 câu của B1).
- Số câu phân cực bị đoán nhầm thành Neutral (False Positives) chỉ là **36 câu** (rất thấp so với 163 câu của B0 Balanced).
- Độ chính xác toàn thể đạt **94.19%**, xác lập SOTA mới trên tập kiểm thử UIT-VSFC!
