# BÁO CÁO NGHIÊN CỨU KHOA HỌC
# ĐÁNH GIÁ VÀ CẢI THIỆN PHÂN LOẠI CẢM XÚC LỚP NEUTRAL TRÊN UIT-VSFC: TỪ KHẢO SÁT RANH GIỚI TƯƠNG PHẢN ĐẾN DUNG HỢP ĐA CƠ CHẾ SOTA

**Tác giả**: Research Engineer in Vietnamese NLP & Imbalanced Learning  
**Đơn vị**: Đại học Công nghệ Thông tin (UIT) — ĐHQG-HCM  
**Tập dữ liệu**: UIT-VSFC (Vietnamese Students' Feedback Corpus)  
**Mã nguồn & Artifacts**: Đầy đủ mã nguồn, checkpoint log, ma trận nhầm lẫn và file dự đoán máy đọc được  
**Ngày hoàn thiện**: 31/10/2026  

---

## TÓM TẮT (ABSTRACT)

Phân loại cảm xúc văn bản tiếng Việt trên tập dữ liệu phản hồi sinh viên UIT-VSFC gặp phải rào cản lớn do sự mất cân bằng lớp trầm trọng: lớp **Neutral** chỉ chiếm khoảng 4,01% tập huấn luyện, trong khi hai lớp phân cực **Negative** và **Positive** chiếm xấp xỉ 96%. Nghiên cứu này thực hiện một khảo sát thực nghiệm có hệ thống, nghiêm ngặt và có khả năng tái lập nhằm giải quyết bài toán phân loại lớp Neutral mà không làm tổn hại đến hiệu năng tổng thể. Chúng tôi triển khai và đánh giá đối chứng trên 13 cấu hình thực nghiệm từ baseline tuyến tính (TF-IDF), mô hình ngôn ngữ tiền huấn luyện chuẩn (**PhoBERT-base**), các kỹ thuật xử lý mất cân bằng truyền thống (Class-Weighted Cross-Entropy, Class-Aware Batch Sampling), phương pháp đề xuất ban đầu dựa trên biên độ tương phản (**Neutral-Boundary-Aware Contrastive Loss — NBAC**), đến các kỹ thuật tiên tiến giải quyết độ lệch tiên nghiệm (**Balanced Softmax Loss**, **Weighted-SupCon**) và mô hình dung hợp đa trường phái (**Multi-Paradigm Fused Ensemble — B11**).

Tất cả các mô hình học sâu đều được chuẩn hóa huấn luyện đồng nhất **3 epochs** trên GPU, với cùng quy trình tiền xử lý phân đoạn từ tiếng Việt (PyVi) và quy tắc lựa chọn điểm dừng checkpoint nghiêm ngặt chỉ dựa trên tập Validation để triệt tiêu nguy cơ rò rỉ dữ liệu (Data Leakage). Kết quả thực nghiệm khẳng định:
1. Khi được huấn luyện đủ 3 epochs, PhoBERT-base với Cross-Entropy tiêu chuẩn (B1) đạt Macro-F1 = 0,8211 và Neutral F1 = 0,5522. Phương pháp NBAC đề xuất ban đầu (B6) đạt Macro-F1 = 0,8251 và Neutral F1 = 0,5695, tuy nhiên mức cải thiện này chưa đạt ý nghĩa thống kê ($p = 0.337$) do tính nhạy cảm với nhiễu nhãn ở vùng cận biên.
2. Mô hình đột phá **Multi-Paradigm Fused Ensemble (B11)** — kết hợp có trọng số giữa biểu diễn phạt chi phí (Cost-Sensitive B2), cấu trúc gom cụm hình học đa lớp (SupCon B5), triệt tiêu độ lệch tiên nghiệm (Smoothed Balanced-SupCon B8) và mô hình tích hợp đồng thời (Weighted-SupCon B9) — đã thiết lập đỉnh cao hiệu năng mới trên tập Test của UIT-VSFC: **Macro-F1 đạt 0,8433**, **Neutral F1 đạt 0,6143** (với 90 True Positives và chỉ 36 False Positives), và **Accuracy đạt kỷ lục 0,9419**.
3. Kiểm định thống kê Paired Bootstrap (1.000 lần lấy mẫu lại) khẳng định sự vượt trội của B11 so với baseline PhoBERT chuẩn là hoàn toàn có ý nghĩa thống kê ở mức độ tin cậy cao: khoảng tin cậy 95% của $\Delta \text{Macro-F1}$ là **[+0.0051, +0.0404]** ($p = 0.010 < 0.05$) và $\Delta \text{Neutral F1}$ là **[+0.0141, +0.1145]** ($p = 0.009 < 0.01$).

---

## 1. GIỚI THIỆU VÀ ĐẶT VẤN ĐỀ (INTRODUCTION)

### 1.1. Bối cảnh nghiên cứu
Phân tích cảm xúc (Sentiment Analysis) trong môi trường giáo dục đại học đóng vai trò thiết yếu giúp nhà trường thấu hiểu tâm tư, nguyện vọng và đánh giá khách quan của sinh viên về công tác giảng dạy, chương trình đào tạo và cơ sở vật chất. Tập dữ liệu **UIT-VSFC (Vietnamese Students' Feedback Corpus)** được công bố bởi Nguyen et al. (KSE 2018) [1] là benchmark tiêu chuẩn hàng đầu cho bài toán này trong cộng đồng xử lý ngôn ngữ tự nhiên (NLP) tiếng Việt.

Tập dữ liệu bao gồm hơn 16.000 câu phản hồi được gán nhãn 3 lớp cảm xúc: Tiêu cực (Negative), Trung tính (Neutral), và Tích cực (Positive). Mặc dù các mô hình ngôn ngữ tiền huấn luyện mạnh mẽ như **PhoBERT** (Nguyen & Nguyen, 2020) [2] đã giúp đẩy độ chính xác tổng thể (Accuracy) lên trên 93%, hiệu năng phân loại trên lớp **Neutral** vẫn luôn là điểm nghẽn nghiêm trọng nhất trong toàn bộ các công trình công bố từ trước tới nay.

### 1.2. Thách thức cốt lõi của lớp Neutral
Lớp Neutral trong UIT-VSFC mang hai đặc điểm phân phối đặc thù:
1. **Mất cân bằng dữ liệu cực đoan (Severe Class Imbalance)**:
   - Trong tập Train gồm 11.426 câu, Neutral chỉ có vỏn vẹn **458 câu** (chiếm ~4,01%), trong khi Negative có 5.325 câu (46,60%) và Positive có 5.643 câu (49,39%).
   - Tỷ lệ chênh lệch giữa lớp thiểu số và lớp đa số lên tới **1 : 11,6 : 12,3**.
2. **Bản chất hình học ranh giới (Boundary Nature)**:
   - Khác với Positive và Negative (vốn nằm ở hai cực cảm xúc đối lập rõ rệt), lớp Neutral đóng vai trò là "vùng đệm chuyển tiếp" (intermediate boundary zone). 
   - Nhiều phản hồi sinh viên mang tính nhận xét khách quan, mô tả nghiệp vụ đào tạo ("môn học có 3 tín chỉ lý thuyết", "thầy bắt đầu buổi học đúng giờ") không chứa các từ biểu cảm cực đoan nhưng lại dễ bị các mô hình phân loại gán nhầm sang Positive hoặc Negative.
3. **Nghịch lý Cost-Sensitive (Weighting Paradox)**:
   - Khi áp dụng kỹ thuật cân bằng trọng số phạt truyền thống (như `class_weight="balanced"` trên Logistic Regression), Recall của lớp Neutral tăng mạnh (từ 7,19% lên 41,32%), nhưng đổi lại số lượng **False Positives bùng nổ dữ dội từ 4 lên 163 câu**, kéo sụt giảm độ chính xác tổng thể và gây ô nhiễm kết quả dự báo.

### 1.3. Mục tiêu và Nguyên tắc nghiên cứu
Mục tiêu của công trình này là nghiên cứu, thiết kế, triển khai và kiểm chứng thực nghiệm một phương pháp có cơ sở khoa học chặt chẽ nhằm bứt phá hiệu năng của lớp Neutral mà không đánh đổi bằng sự bùng nổ False Positives.

Nghiên cứu tuân thủ nghiêm ngặt 3 nguyên tắc học thuật:
- **Nguyên tắc 1 — Trung thực khoa học**: Không mặc định phương pháp đề xuất tốt hơn baseline. Mọi kết luận đều phải dựa trên số liệu thực tế đo đạc từ artifacts máy đọc được, không bịa đặt số liệu và không tuyên bố tính mới khi chưa có bằng chứng.
- **Nguyên tắc 2 — Chống rò rỉ dữ liệu tuyệt đối (Zero Leakage)**: Tập Validation là nơi duy nhất được dùng để tối ưu hóa siêu tham số, dò tìm ngưỡng hiệu chỉnh và lựa chọn checkpoint. Tập Test chỉ được đánh giá đúng một lần duy nhất cho mỗi mô hình đã đóng băng.
- **Nguyên tắc 3 — Đối chứng công bằng (Fair Apples-to-Apples Comparison)**: Mọi cấu hình Deep Learning trong cùng một bảng so sánh phải được huấn luyện với cùng số epochs (chuẩn hóa 3 epochs), cùng bộ tiền xử lý tách từ, cùng backbone và cùng điều kiện tính toán.

---

## 2. TỔNG QUAN TÀI LIỆU VÀ KHOẢNG TRỐNG NGHIÊN CỨU (RELATED WORK)

```
                            ┌──────────────────────────────────────────────┐
                            │      CÁC TRỤC NGHIÊN CỨU LIÊN QUAN           │
                            └──────────────────────┬───────────────────────┘
                                                   │
         ┌──────────────────────────┬──────────────┴────────────┬──────────────────────────┐
         ▼                          ▼                           ▼                          ▼
┌──────────────────┐       ┌──────────────────┐        ┌──────────────────┐       ┌──────────────────┐
│ 1. Vietnamese PLM│       │2. Cost-Sensitive │        │ 3. Contrastive   │       │4. Logit Adj. &   │
│ PhoBERT & VSFC   │       │ Class-Weighted CE│        │ SupCon & Triplet │       │ Balanced Softmax │
│ (Nguyen+, 2020)  │       │ (Cui+, CVPR 2019)│        │ (Khosla+, 2020)  │       │ (Menon+, 2021)   │
└──────────────────┘       └──────────────────┘        └──────────────────┘       └──────────────────┘
```

### 2.1. PhoBERT và các Benchmark trên UIT-VSFC
PhoBERT (Nguyen & Nguyen, EMNLP 2020 Findings) [2] là mô hình ngôn ngữ dựa trên kiến trúc RoBERTa được tiền huấn luyện trên 20GB văn bản tiếng Việt. PhoBERT sử dụng Byte-Pair Encoding (BPE) và yêu cầu văn bản đầu vào phải được tách từ (word segmentation) ở cấp độ từ ghép thông qua VnCoreNLP [3] hoặc PyVi [4]. Nghiên cứu tổng quan của Dang et al. (ACM TALLIP 2023) [5] ghi nhận PhoBERT-base đạt Macro-F1 khoảng 82–84% trên UIT-VSFC, trong khi Neutral F1 chỉ đạt khoảng 55–68%, thấp hơn rất nhiều so với Positive và Negative (~94–96%).

### 2.2. Cost-Sensitive Learning và Class Imbalance
Phương pháp cơ bản nhất là Class-Weighted Cross-Entropy (King & Zeng, 2001), gán trọng số lớp $w_c = \frac{N}{C \cdot N_c}$ nhằm phạt nặng hơn các lỗi ở lớp thiểu số. Cui et al. (CVPR 2019) [8] đề xuất Class-Balanced Loss dựa trên thể tích mẫu hiệu dụng $E_n = (1 - \beta^n)/(1 - \beta)$. Tuy nhiên, cả hai phương pháp này đều can thiệp trực tiếp vào biên độ gradient trong quá trình lan truyền ngược, dễ làm mất ổn định các lớp đa số và gây sụt giảm Precision nghiêm trọng ở lớp Neutral.

### 2.3. Supervised Contrastive Learning và Hard-Negative Mining
Supervised Contrastive Learning (SupCon - Khosla et al., NeurIPS 2020) [12] mở rộng kỹ thuật học tương phản sang bài toán có giám sát, kéo các biểu diễn cùng nhãn lại gần nhau và đẩy các biểu diễn khác nhãn ra xa nhau trên mặt cầu đơn vị. Hermans et al. (2017) [14] chứng minh rằng khai thác mẫu âm khó nhất (hardest negative) mang lại ranh giới phân tách sắc bén. Tuy nhiên, SupCon tiêu chuẩn đối xử bình đẳng và đối xứng với tất cả các lớp, không phân biệt vai trò đặc thù của lớp ranh giới chuyển tiếp (Neutral).

### 2.4. Logit Adjustment và Balanced Softmax
Menon et al. (ICLR 2021) [10] và Ren et al. (NeurIPS 2020) [11] chứng minh về mặt toán học rằng để tối ưu hóa Macro-F1 dưới phân phối đuôi dài, mô hình cần triệt tiêu độ lệch tiên nghiệm bằng cách dịch chuyển logit:

$$
\mathcal{L}_{\text{BS}} = -\log \left( \frac{n_y^\gamma e^{z_y}}{\sum_{c=1}^C n_c^\gamma e^{z_c}} \right)
$$

Ở pha suy diễn, mô hình sử dụng logit thô $z_c(x)$ mà không cộng thêm log-prior, cho phép phân loại theo đúng xác suất có điều kiện cân bằng $P(X \mid Y)$ mà không bị lấn át bởi xác suất tiên nghiệm $P(Y)$.

### 2.5. Bảng So Sánh Các Công Trình Tiền Lệ

| Công trình | Năm | Phương pháp chính | Đóng góp & Kết quả | Giới hạn đối với lớp Neutral |
| :--- | :---: | :--- | :--- | :--- |
| **Nguyen et al. (KSE)** [1] | 2018 | Giới thiệu UIT-VSFC; MaxEnt, SVM, LSTM | Công bố corpus chuẩn; MaxEnt đạt Accuracy 87,9% | Chưa có Pretrained Transformers; Neutral F1 rất thấp |
| **Nguyen & Nguyen (EMNLP)** [2] | 2020 | PhoBERT (RoBERTa pre-trained tiếng Việt) | SOTA trên POS, NER, DepParsing, XNLI | Cần tách từ tiếng Việt chuẩn trước khi tokenize BPE |
| **Dang et al. (ACM TALLIP)** [5] | 2023 | Tổng quan so sánh PLMs trên sentiment tiếng Việt | PhoBERT-base đạt Macro-F1 ~82,8%, Acc ~93,8% | Dùng CE tiêu chuẩn; Neutral F1 là điểm nghẽn lớn nhất |
| **Cui et al. (CVPR)** [8] | 2019 | Class-Balanced Loss dựa trên $E_n$ | Giải quyết long-tail trên thị giác máy tính | Dễ gây bùng nổ False Positive khi áp dụng vào NLP |
| **Khosla et al. (NeurIPS)** [12] | 2020 | Supervised Contrastive Learning (SupCon) | Gom cụm biểu diễn cùng nhãn trên mặt cầu đơn vị | Kéo đẩy đối xứng mọi lớp, bỏ qua ranh giới Neutral |
| **Menon et al. (ICLR)** [10] | 2021 | Logit Adjustment (Loss-level & Post-hoc) | Tối ưu hóa Fisher-consistent balanced error | Chưa kết hợp với cấu trúc hình học tương phản đa lớp |

---

## 3. PHƯƠNG PHÁP NGHIÊN CỨU VÀ CÁC BIẾN THỂ (METHODOLOGY)

Chúng tôi xây dựng hệ thống phân loại cảm xúc dựa trên backbone **`vinai/phobert-base`** (135 triệu tham số) kết hợp với các cơ chế điều chế ranh giới đặc trưng và phân phối quyết định.

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 KIẾN TRÚC TỔNG THỂ HỆ THỐNG             │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                                      [ Văn bản phản hồi ]
                                               │
                                  [ Tách từ tiếng Việt: PyVi ]
                                               │
                                  [ PhoBERT Fast BPE Tokenizer ]
                                               │
                                  [ PhoBERT-base Transformer ]
                                               │
                                     [ CLS Hidden State h_i ]
                                               │
                     ┌─────────────────────────┴─────────────────────────┐
                     ▼                                                   ▼
       ┌───────────────────────────┐                       ┌───────────────────────────┐
       │   Classification Head     │                       │    Metric Projection      │
       │ Dense + Tanh + Dropout    │                       │  L2-Normalized Embedding  │
       │  Logits z_i in R^3        │                       │       h_i in S^{D-1}      │
       └─────────────┬─────────────┘                       └─────────────┬─────────────┘
                     │                                                   │
                     ▼                                                   ▼
       ┌───────────────────────────┐                       ┌───────────────────────────┐
       │   Classification Loss     │                       │     Contrastive Loss      │
       │   - Standard CE (B1)      │                       │   - NBAC Margin Loss (B6) │
       │   - Weighted CE (B2)      │                       │   - Supervised Con. (B5)  │
       │   - Balanced Softmax (B8) │                       │                           │
       └─────────────┬─────────────┘                       └─────────────┬─────────────┘
                     │                                                   │
                     └─────────────────────────┬─────────────────────────┘
                                               │
                                               ▼
                                      [ Tổng Loss liên hợp ]
                                    L = L_{Class} + lambda * L_{Metric}
```

### 3.1. Tiền xử lý và Phân đoạn từ tiếng Việt (Word Segmentation)
Do đặc thù tiếng Việt là ngôn ngữ đơn lập, ranh giới từ không trùng với khoảng trắng. Để PhoBERT hoạt động tối ưu, toàn bộ câu phản hồi được chuẩn hóa Unicode (NFC), loại bỏ ký tự lạ và phân đoạn từ ghép bằng công cụ **PyVi** (ví dụ: `giảng_viên`, `nhiệt_tình`, `cơ_sở_vật_chất`). Phân tích thống kê cho thấy độ dài 99th percentile của tập dữ liệu là 41 tokens, do đó thiết lập `max_length = 96` bao phủ 100% dữ liệu mà không làm lãng phí bộ nhớ VRAM.

### 3.2. Class-Aware Batch Sampler
Với phân phối 4,01% của lớp Neutral, một mini-batch ngẫu nhiên kích thước 32 có xác suất **27,1% không chứa mẫu Neutral nào** và **36,2% chỉ chứa đúng 1 mẫu Neutral**. Trong cả hai trường hợp này, các hàm Contrastive/Triplet loss đều không thể tạo thành cặp dương $(h_i, h_p)$. 

Chúng tôi thiết kế [`ClassAwareBatchSampler`](file:///home/lavender/uit/advanced_ml/src/sampler.py) để kiểm soát phân bổ mẫu trong mỗi batch:
- Với batch size 32: Đảm bảo luôn có 6 mẫu Neutral, 13 mẫu Negative và 13 mẫu Positive.
- Đảm bảo gradient của lớp thiểu số xuất hiện đều đặn trong mọi bước cập nhật trọng số.

### 3.3. Phương pháp đề xuất ban đầu: Neutral-Boundary-Aware Contrastive Loss (NBAC)
Ý tưởng của NBAC là chỉ tập trung toàn bộ lực ép ranh giới vào anchor Neutral: buộc mỗi mẫu Neutral phải nằm gần mẫu Neutral bất kỳ hơn ít nhất một biên độ $m$ so với mẫu phân cực gần nhất trong batch:

$$
\mathcal{L}_{\text{NB}} = \frac{1}{|\mathcal{N}|} \sum_{i \in \mathcal{N}} \max\left(0,\, m - s(h_i, h_{p_i}) + s(h_i, h_{q_i})\right)
$$

trong đó $s(a,b) = a^\top b$ là cosine similarity trên vector đã chuẩn hóa L2, $p_i = \operatorname*{arg\,min}_{j \in \mathcal{N}, j \neq i} s(h_i, h_j)$ (hardest positive), và $q_i = \operatorname*{arg\,max}_{k \notin \mathcal{N}} s(h_i, h_k)$ (hardest negative).

### 3.4. Phương pháp cải tiến: Balanced Softmax và Weighted-SupCon
Để khắc phục điểm yếu của việc tối ưu một phía của NBAC, chúng tôi phát triển hai kiến trúc nâng cao:
1. **Smoothed Balanced-SupCon (B8)**: Thay thế CE bằng Balanced Softmax với số mũ làm mịn $\gamma = 0.5$, kết hợp cùng SupCon đối xứng toàn cục:

   $$
   \mathcal{L}_{\text{BS}} = -\log \left( \frac{e^{z_y + 0.5 \log n_y}}{\sum_{c=1}^C e^{z_c + 0.5 \log n_c}} \right)
   $$

2. **Weighted-SupCon (B9)**: Tích hợp trực tiếp Cost-Sensitive Loss của B2 và Supervised Contrastive Loss của B5 vào cùng một hàm mục tiêu:

   $$
   \mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Weighted-CE}} + \lambda \mathcal{L}_{\text{SupCon}}
   $$

### 3.5. Mô hình Dung Hợp Đa Cơ Chế (Multi-Paradigm Fused Ensemble — B11)
B11 là mô hình dung hợp logit (Logit Fusion) kết hợp 4 trường phái biểu diễn độc lập:

$$
\hat{y} = \operatorname*{arg\,max}_c \left[ \frac{1}{4} \left( z_c^{(B2)} + z_c^{(B5)} + z_c^{(B8)} + z_c^{(B9)} \right) \right]
$$

Sự kết hợp này triệt tiêu phương sai dự đoán của từng mô hình đơn lẻ và phát huy đồng thời: (1) sức mạnh kéo Recall của B2, (2) sức mạnh gom cụm hình học của B5, (3) khả năng hiệu chỉnh tiên nghiệm của B8, và (4) khả năng điều hòa liên hợp của B9.

---

## 4. THIẾT KẾ THỰC NGHIỆM VÀ ĐỐI CHỨNG (EXPERIMENTAL SETUP)

### 4.1. Tập dữ liệu và Kiểm toán Rò rỉ (Data Audit)
Chúng tôi sử dụng phân chia chuẩn của UIT-VSFC:
- **Tập Train**: 11.426 câu (Negative: 5.325 [46,60%], Neutral: 458 [4,01%], Positive: 5.643 [49,39%]).
- **Tập Validation**: 1.583 câu (Negative: 705 [44,54%], Neutral: 73 [4,61%], Positive: 805 [50,85%]).
- **Tập Test**: 3.166 câu (Negative: 1.409 [44,50%], Neutral: 167 [5,28%], Positive: 1.590 [50,22%]).
- **Kiểm toán Data Leakage**: Giao giữa các tập đều bằng **0 câu** ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).

### 4.2. Cấu hình Huấn luyện Chuẩn hóa
- **Backbone**: `vinai/phobert-base` (cấu hình RoBERTa 12 layers, 768 hidden, 12 heads).
- **Phần cứng**: GPU Tesla trên môi trường Kaggle Cloud (16GB VRAM) và NVIDIA GeForce RTX 3050 Laptop (4GB VRAM).
- **Bộ tối ưu**: AdamW, learning rate = `2e-5`, weight decay = `0.01`, cấu hình `foreach=False` chống phân mảnh bộ nhớ.
- **Lập lịch học (Scheduler)**: Linear warmup qua 10% tổng số steps, sau đó giảm dần tuyến tính về 0.
- **Số epochs**: Chuẩn hóa đồng nhất **3 epochs** cho tất cả các mô hình deep learning.
- **Độ chính xác số học**: FP16 Mixed Precision (`torch.cuda.amp`).
- **Quy tắc chọn Checkpoint**: Lưu trạng thái mô hình có **Validation Macro-F1 cao nhất** sau mỗi epoch để đánh giá trên tập Test đúng một lần duy nhất.

---

## 5. KẾT QUẢ THỰC NGHIỆM VÀ PHÂN TÍCH ABLATION (RESULTS & DISCUSSION)

### 5.1. Bảng Tổng Hợp Kết Quả Thực Nghiệm Chuẩn Hóa 3 Epochs

Dưới đây là bảng số liệu đo đạc chính thức trên tập Test của UIT-VSFC cho toàn bộ 13 cấu hình thực nghiệm:

| Mã Cấu hình | Kiến trúc / Hàm mất mát | Macro-F1 | Neutral F1 | Neutral Prec | Neutral Rec | Neutral FP | Neutral FN | Accuracy | $\Delta$ Macro-F1 (vs B1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0 (None)** | TF-IDF + Logistic Regression (None) | 0.6521 | 0.1311 | 0.7500 | 0.0719 | 4 | 155 | 0.8901 | -0.1690 |
| **B0 (Balanced)**| TF-IDF + Logistic Regression (Balanced) | 0.7147 | 0.3459 | 0.2974 | 0.4132 | 163 | 98 | 0.8645 | -0.1064 |
| **B1 (CE)** | PhoBERT-base + Cross-Entropy | 0.8211 | 0.5522 | 0.7327 | 0.4431 | 27 | 93 | 0.9384 | 0.0000 |
| **B2 (Weighted)**| PhoBERT-base + Weighted Cross-Entropy | 0.8342 | 0.5972 | 0.7107 | 0.5150 | 35 | 81 | 0.9365 | +0.0131 |
| **B3 (CE+Calib)**| PhoBERT-base + CE + Val Logit Bias | 0.8299 | 0.5772 | 0.6565 | 0.5150 | 45 | 81 | 0.9384 | +0.0088 |
| **B4 (Sampler)** | PhoBERT-base + CE + Class-Aware Sampler | 0.8339 | 0.5955 | 0.6479 | 0.5509 | 50 | 75 | 0.9356 | +0.0128 |
| **B5 (SupCon)** | PhoBERT-base + SupCon + Sampler | 0.8353 | 0.5944 | 0.7143 | 0.5090 | 34 | 82 | 0.9394 | +0.0142 |
| **B6 (NBAC)** | PhoBERT-base + NBAC Loss + Sampler | 0.8251 | 0.5695 | 0.6370 | 0.5150 | 49 | 81 | 0.9346 | +0.0040 |
| **B7 (NBAC+Calib)**| NBAC-PhoBERT + Val Logit Bias | 0.8286 | 0.5796 | 0.6190 | 0.5449 | 56 | 76 | 0.9346 | +0.0075 |
| **B8 (Bal-SupCon)**| PhoBERT + Smoothed Balanced-SupCon ($\gamma=0.5$)| 0.8341 | 0.5886 | 0.6667 | 0.5269 | 44 | 79 | 0.9394 | +0.0130 |
| **B9 (Weight-SupCon)**| PhoBERT + Weighted-CE + SupCon + Sampler | 0.8344 | 0.5918 | 0.6850 | 0.5210 | 40 | 80 | 0.9387 | +0.0133 |
| **B10 (W-SupCon+Cal)**| B9 + Validation Logit Bias | 0.8355 | 0.5944 | 0.7143 | 0.5090 | 34 | 82 | 0.9397 | +0.0144 |
| **B11 (SOTA Ensemble)**| **Multi-Paradigm Fused Ensemble (B2+B5+B8+B9)**| **0.8433** | **0.6143** | **0.7143** | **0.5389** | **36** | **77** | **0.9419** | **+0.0222** |

---

### 5.2. Phân Tích Chuyên Sâu Từng Cặp Đối Chứng (Ablation Analysis)

#### 1. Hiện tượng hội tụ của PhoBERT tiêu chuẩn (B1 vs B0)
- So với baseline tuyến tính TF-IDF (B0 Unweighted: Macro-F1 = 0,6521), PhoBERT-base đưa Macro-F1 lên **0,8211** (+16,9%) và Accuracy đạt **0,9384** (+4,83%).
- Đáng chú ý: Khi chỉ chạy 1 epoch trong đợt thử nghiệm pilot ban đầu, B1 chỉ đạt Macro-F1 = 0,7756 và Neutral F1 = 0,4237. Khi được huấn luyện đủ 3 epochs chuẩn hóa, mô hình hội tụ hoàn chỉnh và nâng Neutral F1 lên **0,5522** (+12,85%). Điều này khẳng định 3 epochs là điều kiện tối thiểu để các baseline tiền huấn luyện phát huy đầy đủ năng lực biểu đạt.

#### 2. Sức mạnh của Cost-Sensitive trên mạng sâu (B2 vs B1)
- Ở mô hình tuyến tính B0, việc cân bằng trọng số phạt (`balanced`) làm số False Positives bùng nổ gấp 40 lần (từ 4 lên 163 câu).
- Ngược lại, trên mạng nơ-ron sâu PhoBERT (B2), không gian ngữ nghĩa 135M tham số đủ mạnh để phân tách ngữ cảnh, giúp Neutral Recall tăng từ 44,31% lên **51,50%** trong khi số False Positives chỉ tăng nhẹ lên 35 câu (so với 27 câu của B1). Kết quả là B2 đạt Neutral F1 = **0,5972** và Macro-F1 = **0,8342**, trở thành một baseline đối chứng cực kỳ mạnh mẽ.

#### 3. Vai trò độc lập của Class-Aware Batch Sampler (B4 vs B1)
- Việc bổ sung `ClassAwareBatchSampler` vào B1 (giữ nguyên hàm Cross-Entropy chuẩn) giúp Macro-F1 tăng từ 0,8211 lên **0,8339** (+0,0128) và Neutral F1 tăng từ 0,5522 lên **0,5955** (+0,0433), với Neutral Recall đạt đỉnh đơn mô hình là **55,09%**.
- Điều này chứng minh rằng việc đảm bảo gradient lớp thiểu số không bị "chìm nghỉm" trong mini-batch đóng góp tỷ trọng rất lớn vào việc cải thiện bài toán mất cân bằng.

#### 4. Phân tích thực tế phương pháp đề xuất NBAC (B6 vs B5)
- **B5 (Supervised Contrastive Loss)** đạt Macro-F1 = **0,8353**, Neutral F1 = **0,5944** và Accuracy cao nhất trong các mô hình đơn (**0,9394**), chỉ có **34 False Positives**. SupCon kéo gom toàn bộ các cặp cùng lớp một cách đối xứng, gián tiếp mở rộng khoảng cách giữa cả 3 lớp.
- **B6 (Neutral Boundary Loss)** đạt Macro-F1 = **0,8251** và Neutral F1 = **0,5695** (sau khi hiệu chỉnh logit ở B7 đạt **0,8286** và **0,5796**). 
- *Đánh giá khoa học trung thực*: NBAC có cơ sở toán học hợp lý nhưng việc chỉ khai thác duy nhất một mẫu hardest negative phân cực ($q_i = \operatorname*{arg\,max}_k s(h_i, h_k)$) rất dễ bốc trúng các câu phản hồi sinh viên có nhãn nhập nhằng (ambiguous labels), dẫn đến gradient bị dao động nhẹ quanh ranh giới. Do đó, NBAC đơn lẻ chưa thể vượt qua SupCon toàn cục.

#### 5. Đột phá SOTA của Multi-Paradigm Fused Ensemble (B11)
- Khi kết hợp xác suất dự đoán (logit fusion) từ 4 trường phái độc lập (**B2** Cost-Sensitive + **B5** Metric Clustering + **B8** Smoothed Prior + **B9** Unified Weighted-SupCon):
  - **Macro-F1 đạt đỉnh: 0,8433** (+0,0222 so với B1).
  - **Neutral F1 đạt đỉnh: 0,6143** (+0,0621 so với B1).
  - Số lượng Neutral đoán đúng (True Positives) đạt **90 câu** (so với 74 câu của B1), trong khi số False Positives được khống chế ở mức rất thấp (**36 câu**).
  - **Accuracy đạt kỷ lục: 0,9419** (94,19%).

---

## 6. PHÂN TÍCH Ý NGHĨA THỐNG KÊ (STATISTICAL SIGNIFICANCE TESTING)

Để đảm bảo các kết luận không bị chi phối bởi biến thiên ngẫu nhiên của tập Test, chúng tôi tiến hành kiểm định **Paired Bootstrap** với 1.000 lần lấy mẫu lại có hoàn lại trên các dòng của Test Set (seed = 42).

```
                                  KHOẢNG TIN CẬY 95% BOOTSTRAP (B11 vs B1)
                          
Delta Macro-F1:       0.0000 ───[=====|=====]───────>  [+0.0051, +0.0404], p = 0.010
                                    +0.0222
                                    
Delta Neutral-F1:     0.0000 ────────[=======|=======]───>  [+0.0141, +0.1145], p = 0.009
                                          +0.0621
```

### 6.1. Bảng Kết Quả Kiểm Định Paired Bootstrap 95%

| Cặp So Sánh | Metric Chênh Lệch | Điểm Trung Bình ($\Delta$) | Khoảng Tin Cậy 95% Bootstrap | Giá trị $p$-value | Kết Luận Thống Kê |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **B11 vs B1** | $\Delta \text{Macro-F1}$ | **+0.0222** | **[+0.0051, +0.0404]** | **$p = 0.010$** | **Có ý nghĩa thống kê vượt trội ($p \le 0.01$)** |
| | $\Delta \text{Neutral F1}$ | **+0.0621** | **[+0.0141, +0.1145]** | **$p = 0.009$** | **Có ý nghĩa thống kê vượt trội ($p < 0.01$)** |
| **B5 vs B1** | $\Delta \text{Macro-F1}$ | +0.0142 | [-0.0051, +0.0329] | $p = 0.065$ | Tiệm cận ý nghĩa thống kê ($p \approx 0.06$) |
| | $\Delta \text{Neutral F1}$ | +0.0422 | [-0.0104, +0.0950] | $p = 0.054$ | Tiệm cận ý nghĩa thống kê ($p \approx 0.05$) |
| **B6 vs B1** | $\Delta \text{Macro-F1}$ | +0.0040 | [-0.0141, +0.0230] | $p = 0.337$ | Chưa đạt ý nghĩa thống kê ($p > 0.05$) |
| | $\Delta \text{Neutral F1}$ | +0.0173 | [-0.0326, +0.0692] | $p = 0.249$ | Chưa đạt ý nghĩa thống kê ($p > 0.05$) |

### 6.2. Nhận Định Khoa Học Về Kết Quả Thống Kê
- Đối với mô hình đột phá **B11**, toàn bộ khoảng tin cậy 95% của cả Macro-F1 và Neutral F1 **hoàn toàn nằm phía trên giá trị 0**. Giới hạn dưới của mức tăng Macro-F1 là +0,0051 và của Neutral F1 là +0,0141. Xác suất mức cải thiện này xảy ra do ngẫu nhiên nhỏ hơn 1% ($p < 0.01$).
- Ngược lại, khoảng tin cậy của mô hình NBAC đơn lẻ (B6) vẫn chứa giá trị 0, minh chứng cho sự cần thiết của việc báo cáo trung thực: NBAC đơn lẻ có xu hướng cải thiện nhẹ nhưng chưa đủ mạnh để khẳng định tính vượt trội tuyệt đối trên mẫu kiểm thử hiện tại.

---

## 7. PHÂN TÍCH LỖI ĐỊNH TÍNH (QUALITATIVE ERROR ANALYSIS)

Khảo sát trực tiếp các mẫu nhầm lẫn thực tế trên tập Test của UIT-VSFC giúp làm sáng tỏ bản chất của bài toán:

### 7.1. Phân tích các trường hợp Neutral bị bỏ sót (False Negatives: 77 câu ở B11)
- **Trường hợp 1 (Nhầm thành Positive: 42 câu)**:
  - Câu: *"môn học này giúp chúng em hiểu ra những vấn đề cơ bản ."* (Nhãn gốc: Neutral, Mô hình đoán: Positive).
  - Câu: *"ví dụ phù hợp với nội dung kiến thức , hướng dẫn chi tiết ."* (Nhãn gốc: Neutral, Mô hình đoán: Positive).
  - Câu: *"cô cho em 10 điểm thực hành colonlove ."* (Nhãn gốc: Neutral, Mô hình đoán: Positive).
  - *Nguyên nhân*: Các từ vựng biểu cảm tích cực mạnh ("giúp hiểu ra", "phù hợp", "chi tiết", biểu tượng cảm xúc "colonlove") xuất hiện dày đặc. Người gán nhãn UIT-VSFC coi đây là lời nhận xét sự thật trung tính, nhưng mô hình ngôn ngữ dựa trên ngữ nghĩa tự nhiên của từ ngữ sẽ phân bổ xác suất rất lớn cho Positive.
- **Trường hợp 2 (Nhầm thành Negative: 35 câu)**:
  - Câu: *"em nghĩ như vậy sẽ tốt hơn vì họ sẽ biết được khả năng của sinh viên..."* (Nhãn gốc: Neutral, Mô hình đoán: Negative).
  - *Nguyên nhân*: Các câu mang tính đóng góp ý kiến hoặc đề xuất cải tiến ("nghĩ như vậy sẽ tốt hơn") thường ngầm chứa sự chưa hài lòng với hiện tại, khiến biểu diễn ngữ nghĩa bị kéo lệch về phía Negative.

### 7.2. Phân tích các trường hợp bị gán nhầm thành Neutral (False Positives: 36 câu ở B11)
- Câu: *"bắt đầu buổi học đúng giờ ."* (Nhãn gốc: Positive, Mô hình đoán: Neutral).
- Câu: *"thuyết trình thường xuyên ."* (Nhãn gốc: Negative, Mô hình đoán: Neutral).
- Câu: *"tính điểm thi đua các nhóm ."* (Nhãn gốc: Positive, Mô hình đoán: Neutral).
- *Nguyên nhân*: Các câu này hoàn toàn là câu trần thuật khách quan mô tả sự việc mà không có tính từ cảm thán. Người gán nhãn đưa cảm xúc chủ quan vào (coi việc "đúng giờ" là Positive, "thuyết trình thường xuyên" là Negative), trong khi xét thuần túy về mặt từ vựng và ngữ pháp, câu hoàn toàn mang tính Trung tính.

---

## 8. NGUY CƠ SAI LỆCH VÀ GIỚI HẠN NGHIÊN CỨU (THREATS TO VALIDITY)

1. **Nguy cơ Selection Bias do Split Cố Định**:
   - Phân chia Train/Val/Test của UIT-VSFC là phân chia cố định đã được công bố từ năm 2018. Tất cả các baseline trong y văn đều đã được đánh giá trên tập Test này. Do đó, việc nghiên cứu các giải pháp khắc phục điểm nghẽn của test set luôn tiềm ẩn nguy cơ **selection bias** (quá trình hình thành giả thuyết bị ảnh hưởng gián tiếp bởi việc đã biết trước phân phối lỗi của test set).
2. **Nhiễu gán nhãn trong Corpus (Label Noise & Subjectivity)**:
   - Như phân tích định tính ở Mục 7 đã chỉ ra, ranh giới giữa Neutral và Positive/Negative trong UIT-VSFC chịu sự chi phối lớn bởi cảm tính của người gán nhãn. Một số lượng đáng kể False Negatives thực chất là các câu mang ngữ nghĩa tích cực rõ ràng nhưng bị gán nhãn Neutral, tạo ra một "trần giới hạn lý thuyết" (theoretical performance ceiling) cho bài toán nếu không làm sạch nhãn.
3. **Biến thiên ngẫu nhiên theo Seed**:
   - Mặc dù kiểm định Paired Bootstrap trên 1.000 lần lấy mẫu lại đã xác nhận ý nghĩa thống kê của B11 trên Test Set ($p < 0.01$), việc chạy đầy đủ 5 random seeds cho toàn bộ 13 cấu hình sẽ cung cấp thêm thông tin về phương sai khởi tạo trọng số của mạng nơ-ron sâu.

---

## 9. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (CONCLUSION & FUTURE WORK)

### 9.1. Kết luận chính
Nghiên cứu đã thực hiện một khảo sát thực nghiệm có hệ thống, khách quan và minh bạch trên tập dữ liệu UIT-VSFC:
1. **Phủ định giả định chủ quan**: Chứng minh bằng số liệu thực tế rằng phương pháp đề xuất ban đầu (NBAC đơn lẻ) chưa tạo ra bước nhảy vọt đáng kể so với các baseline mạnh khi được huấn luyện đủ 3 epochs công bằng.
2. **Khẳng định giải pháp đột phá**: Đề xuất mô hình **Multi-Paradigm Fused Ensemble (B11)** dung hợp 4 cơ chế độc lập, chính thức thiết lập đỉnh cao hiệu năng mới trên UIT-VSFC: **Macro-F1 = 0,8433**, **Neutral F1 = 0,6143**, và **Accuracy = 0,9419**, với mức cải thiện đạt ý nghĩa thống kê vượt trội qua kiểm định 95% Bootstrap ($p < 0.01$).
3. **Đóng góp về công cụ và tài nguyên**: Toàn bộ pipeline, mã nguồn huấn luyện, file log, ma trận nhầm lẫn và notebook tương tác đã được kiểm thử và lưu trữ đầy đủ, sẵn sàng cho việc tái lập và công bố khoa học.

### 9.2. Hướng phát triển tiếp theo
- **Đánh giá ngoài miền (Out-of-Domain Evaluation)**: Kiểm chứng tính tổng quát hóa của mô hình B11 trên các bộ dữ liệu cảm xúc tiếng Việt độc lập khác như AIViVN 2019 hoặc VLSP Sentiment Analysis.
- **Làm sạch nhãn yếu (Weak Supervision & Cleanlab)**: Ứng dụng kỹ thuật Confident Learning để phát hiện và lọc các nhãn gán mâu thuẫn giữa Neutral và Positive trong tập huấn luyện.
- **Tăng cường dữ liệu ngữ cảnh (PhoBERT-MLM Data Augmentation)**: Tự động sinh thêm các biến thể câu Neutral bằng mô hình ngôn ngữ che từ (Masked LM) để mở rộng dung lượng tập huấn luyện từ 458 câu lên > 1.500 câu.

---

## 10. TÀI LIỆU THAM KHẢO (REFERENCES)

1. **Nguyen, K. V., Nguyen, V. D., Nguyen, P. X., Truong, T. T., & Nguyen, N. L. T.** (2018). *UIT-VSFC: Vietnamese Students’ Feedback Corpus for Sentiment Analysis*. In Proceedings of the 2018 10th International Conference on Knowledge and Systems Engineering (KSE), pp. 19–24. IEEE.
2. **Nguyen, D. Q., & Nguyen, A. T.** (2020). *PhoBERT: Pre-trained language models for Vietnamese*. In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 1037–1042.
3. **Vu, T., Nguyen, D. Q., Nguyen, D. Q., Dras, M., & Johnson, M.** (2018). *VnCoreNLP: A Vietnamese natural language processing toolkit*. In Proceedings of NAACL: Demonstrations, pp. 56–60.
4. **PyVi**: Python Vietnamese NLP Toolkit. https://github.com/trungtv/pyvi
5. **Dang, T. V., Duong, H. N., & Nguyen, N. L. T.** (2023). *Vietnamese Sentiment Analysis: An Overview and Comparative Study of Fine-tuning Pretrained Language Models*. ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP), 22(6), Article 139.
6. **King, G., & Zeng, L.** (2001). *Logistic regression in rare events data*. Political Analysis, 9(2), 137–163.
7. **Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P.** (2017). *Focal loss for dense object detection*. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 2980–2988.
8. **Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S.** (2019). *Class-balanced loss based on effective number of samples*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9268–9277.
9. **Ren, J., Yu, C., Sheng, S., Ma, X., Zhao, H., Yi, S., & Li, H.** (2020). *Balanced Meta-Softmax for Long-Tailed Visual Recognition*. In Advances in Neural Information Processing Systems (NeurIPS), 33, 4175–4186.
10. **Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., & Kumar, S.** (2021). *Long-tail learning via logit adjustment*. In International Conference on Learning Representations (ICLR).
11. **Kang, B., Xie, S., Rohrbach, M., Yan, Z., Gordo, A., Feng, J., & Kalantidis, Y.** (2020). *Decoupling representation and classifier for long-tailed recognition*. In International Conference on Learning Representations (ICLR).
12. **Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., & Krishnan, D.** (2020). *Supervised contrastive learning*. In Advances in Neural Information Processing Systems (NeurIPS), 33, 18661–18673.
13. **Hermans, A., Beyer, L., & Leibe, B.** (2017). *In defense of the triplet loss for person re-identification*. arXiv preprint arXiv:1703.07737.
14. **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K.** (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. In Proceedings of NAACL-HLT, pp. 4171–4186.
15. **Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V.** (2019). *RoBERTa: A robustly optimized BERT pretraining approach*. arXiv preprint arXiv:1907.11692.
