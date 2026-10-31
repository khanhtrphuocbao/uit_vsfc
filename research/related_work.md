# TỔNG QUAN TÀI LIỆU VÀ CƠ SỞ KHOA HỌC (RELATED WORK)

Tài liệu này tổng hợp cơ sở lý thuyết, các công trình gốc, các benchmark công bố và khoảng trống nghiên cứu liên quan đến bài toán phân loại cảm xúc trên tập dữ liệu UIT-VSFC (Vietnamese Students' Feedback Corpus), đặc biệt tập trung vào thách thức phân loại lớp mất cân bằng **Neutral**.

---

## 1. Các Trục Nghiên Cứu Chính

### 1.1. Dataset UIT-VSFC và Pretrained Language Models tiếng Việt

* **Tập dữ liệu UIT-VSFC**: Được công bố bởi Nguyen et al. (KSE 2018) [1], gồm hơn 16.000 câu phản hồi của sinh viên, chia làm hai nhiệm vụ: phân loại cảm xúc (Sentiment: Negative, Neutral, Positive) và chủ đề (Topic: 4 lớp). Split chuẩn được công bố gồm:
  - **Train**: 11.426 câu (Negative: 5.325 [46,60%], Neutral: 458 [4,01%], Positive: 5.643 [49,39%]).
  - **Validation**: 1.583 câu (Negative: 705 [44,54%], Neutral: 73 [4,61%], Positive: 805 [50,85%]).
  - **Test**: 3.166 câu (Negative: 1.409 [44,50%], Neutral: 167 [5,28%], Positive: 1.590 [50,22%]).
  Lớp **Neutral** là lớp thiểu số trầm trọng (chỉ chiếm ~4,01% tập train và ~5,28% tập test), dẫn đến độ mất cân bằng tỷ lệ xấp xỉ 1 : 11,6 : 12,3.
* **PhoBERT**: Công bố bởi Nguyen & Nguyen (EMNLP 2020 Findings) [2], dựa trên kiến trúc RoBERTa tiền huấn luyện trên 20GB văn bản tiếng Việt. PhoBERT yêu cầu văn bản đầu vào phải được tách từ (word segmentation) ở cấp độ từ ghép (ví dụ: `sinh_viên`, `nhiệt_tình`) thông qua công cụ RDRsegmenter / VnCoreNLP [3] hoặc PyVi [4] trước khi áp dụng BPE tokenization.
* **Benchmark PLM trên UIT-VSFC**: Nghiên cứu tổng quan của Dang et al. (ACM TALLIP 2023) [5] và các công trình liên quan [6, 7] ghi nhận:
  - Các mô hình truyền thống (TF-IDF + MaxEnt, SVM, Logistic Regression) đạt Accuracy khoảng 87–88%, nhưng Macro-F1 chỉ đạt 64–72% do Neutral F1 rất thấp (12–37%).
  - PhoBERT-base fine-tune đạt Accuracy 92–94% và Macro-F1 khoảng 81–84%. Tuy nhiên, điểm yếu cố hữu vẫn nằm ở lớp Neutral: F1 lớp Neutral chỉ dao động quanh 60–78%, thấp hơn rất nhiều so với Positive (~94–96%) và Negative (~93–95%). Nhiều câu Neutral mang tính chất nhận xét trung tính ("thầy dạy bình thường", "môn học có 3 tín chỉ") bị gán nhầm vào Negative hoặc Positive do ngữ cảnh yếu, hoặc ngược lại khi ép trọng số thì False Positive Neutral tăng đột biến.

---

### 1.2. Mất Cân Bằng Dữ Liệu và Cost-Sensitive Learning

* **Class-weighted Cross-Entropy (Balanced Loss)**: Phương pháp cơ bản gán trọng số lớp $w_c = \frac{N}{C \cdot N_c}$ (King & Zeng, 2001) nhằm phạt nặng hơn các lỗi phân loại ở lớp thiểu số.
* **Class-Balanced Loss (CB Loss)**: Cui et al. (CVPR 2019) [8] chỉ ra rằng khi số lượng mẫu tăng, thông tin biên bổ sung giảm dần theo thể tích không gian đặc trưng. Họ đề xuất tính trọng số theo "số lượng mẫu hiệu dụng" (effective number of samples) $E_n = (1 - \beta^n)/(1 - \beta)$ với $\beta \in [0.9, 0.9999]$.
* **Focal Loss**: Lin et al. (ICCV 2017) [9] thêm nhân tử điều chế $(1 - p_t)^\gamma$ để giảm trọng số của các mẫu dễ (easy examples) và tập trung vào các mẫu khó (hard examples).
* **Đặc thù đối với Neutral**: Khi tăng trọng số phạt (Cost-sensitive) trên lớp Neutral, mô hình có xu hướng đẩy ngưỡng quyết định về phía Neutral. Hệ quả là **Neutral Recall tăng**, nhưng **Neutral Precision sụt giảm nghiêm trọng** do sinh ra lượng lớn False Positive (câu phân cực nhẹ hoặc câu dài có chứa từ trung tính bị dự đoán thành Neutral). Điều này lý giải hiện tượng trong bản nháp: TF-IDF với `class_weight="balanced"` tăng Neutral Recall nhưng kéo theo sụt giảm Accuracy và sinh nhiều False Positive.

---

### 1.3. Logit Adjustment và Balanced Softmax

* **Logit Adjustment**: Menon et al. (ICLR 2021) [10] chứng minh về mặt lý thuyết rằng để tối ưu hóa Macro-F1 hoặc balanced error dưới phân phối đuôi dài (long-tail), mô hình cần dịch chuyển logit dự đoán theo phân phối tiên nghiệm của các lớp:
  $$\hat{y} = \arg\max_y \left( f_y(x) - \tau \log \pi_y \right)$$
  trong đó $\pi_y$ là xác suất tiên nghiệm của lớp $y$ trong tập huấn luyện, và $\tau > 0$ là nhiệt độ điều chỉnh.
  - *Post-hoc Logit Adjustment*: Áp dụng sau khi huấn luyện mô hình chuẩn bằng Cross-Entropy mà không làm biến dạng biểu diễn biểu đạt.
  - *Loss-level Logit Adjustment*: Đưa độ lệch tiên nghiệm trực tiếp vào hàm Softmax khi tính loss trong quá trình huấn luyện.
* **Balanced Softmax**: Ren et al. (NeurIPS 2020) [11] đề xuất cách tiếp cận tương đương nhằm giải quyết vấn đề conditional shift trong softmax phân loại long-tail.
* **Ý nghĩa đối với bài toán**: Hiệu chỉnh logit Neutral bằng validation set cung cấp cơ chế kiểm soát trực tiếp trade-off giữa Precision và Recall mà không cần can thiệp thô bạo vào trọng số gradient trong quá trình lan truyền ngược.

---

### 1.4. Contrastive Learning, Hard Negative Mining và Class-Aware Sampling

* **Supervised Contrastive Learning (SupCon)**: Khosla et al. (NeurIPS 2020) [12] mở rộng SimCLR [13] sang bài toán có giám sát, kéo các biểu diễn cùng nhãn lại gần nhau và đẩy các biểu diễn khác nhãn ra xa nhau trên mặt cầu đơn vị.
* **Batch-Hard Triplet Loss**: Hermans et al. (2017) [14] trong bài toán re-identification chứng minh rằng việc khai thác mẫu âm khó nhất (hardest negative: $q = \arg\max_{k \notin y_i} s(h_i, h_k)$) và mẫu dương khó nhất (hardest positive: $p = \arg\min_{j \in y_i} s(h_i, h_j)$) trong mini-batch mang lại khả năng phân định ranh giới (margin) chặt chẽ hơn nhiều so với việc lấy mẫu ngẫu nhiên.
* **Ranh giới lớp Neutral (Neutral Boundary)**: Khác với Positive và Negative (vốn nằm ở hai cực cảm xúc đối lập), Neutral nằm ở "vùng đệm" (boundary zone) giữa hai thái cực. Do đó, một câu Neutral dễ bị nhầm lẫn nhất với những câu Negative hoặc Positive có sắc thái nhẹ hoặc ngữ nghĩa gần gũi.
* **Class-Aware Batch Sampling**: Shen et al. (2016) [15] và Kang et al. (ICLR 2020) [16] chỉ ra rằng trong phân loại đuôi dài, mini-batch ngẫu nhiên sẽ chứa rất ít (hoặc không có) mẫu thiểu số. Với batch size 32 trên UIT-VSFC, xác suất một batch không có mẫu Neutral nào lên tới $(1 - 0.0401)^{32} \approx 27,1\%$, và xác suất chỉ có đúng 1 mẫu Neutral là $\approx 36,2\%$. Trong cả hai trường hợp này, các hàm Contrastive/Triplet loss đều vô hiệu vì không thể tạo thành cặp dương $(h_i, h_{p_i})$. Do đó, Class-aware sampler là điều kiện tiên quyết để contrastive loss hoạt động ổn định.

---

### 1.5. Calibration và Validation Thresholding

* **Confidence Calibration**: Guo et al. (ICML 2017) [17] chỉ ra rằng các mạng nơ-ron sâu hiện đại thường bị tự tin thái quá (overconfident).
* **Validation Thresholding**: Provost (AAAI 2000) [18] và Lipton et al. (NeurIPS 2014) [19] nhấn mạnh rằng việc chọn ngưỡng quyết định hoặc dịch chuyển bias logit phải thực hiện nghiêm ngặt trên tập Validation độc lập để tránh selection bias và data leakage.

---

## 2. Bảng So Sánh Các Công Trình Liên Quan

| Công trình | Năm | Kỹ thuật / Đóng góp | Kết quả trên UIT-VSFC (hoặc benchmark tương đương) | Giới hạn & Khoảng trống nghiên cứu |
| :--- | :---: | :--- | :--- | :--- |
| **Nguyen et al. (KSE)** [1] | 2018 | Giới thiệu corpus UIT-VSFC; baselines MaxEnt, SVM, LSTM. | MaxEnt F1 = 87,94% (accuracy); Neutral F1 thấp, chưa phân tích sâu ranh giới Neutral. | Chưa có pretrained transformer; phân tích chi tiết từng lớp còn hạn chế. |
| **Nguyen & Nguyen (EMNLP)** [2] | 2020 | Giới thiệu PhoBERT-base và PhoBERT-large (RoBERTa pre-trained cho tiếng Việt). | SOTA trên POS, NER, DepParsing, XNLI. Chưa báo cáo chi tiết UIT-VSFC 3-class trong paper gốc. | Cần văn bản tách từ tiếng Việt chuẩn trước khi tokenize. |
| **Dang et al. (ACM TALLIP)** [5] | 2023 | Đánh giá so sánh các PLMs (PhoBERT, ViBERT, XLM-R, v.v.) trên sentiment tiếng Việt. | PhoBERT-base đạt Macro-F1 ~82,8%, Acc ~93,8%. Neutral F1 là điểm nghẽn lớn nhất (~68–74%). | Chỉ dùng Cross-Entropy tiêu chuẩn, chưa giải quyết chuyên sâu class imbalance lớp Neutral. |
| **Cui et al. (CVPR)** [8] | 2019 | Class-Balanced Loss dựa trên số lượng mẫu hiệu dụng ($E_n$). | Cải thiện bài toán long-tail trên CIFAR-LT và ImageNet-LT. | Giảm thiểu mất cân bằng bằng re-weighting nhưng dễ gây bùng nổ False Positive ở lớp thiểu số trung tính. |
| **Menon et al. (ICLR)** [10] | 2021 | Logit Adjustment (post-hoc và loss-level) dịch chuyển logit theo prior $\pi_y$. | Cải thiện balanced accuracy/macro-F1 trên iNaturalist, ImageNet-LT mà không làm méo feature space. | Chưa kết hợp với cơ chế định hình ranh giới biểu diễn (representation boundary shaping) của lớp trung gian. |
| **Khosla et al. (NeurIPS)** [12] | 2020 | Supervised Contrastive Learning (SupCon) cho toàn bộ các lớp. | Vượt qua Cross-Entropy trên phân loại ảnh và nhận diện mẫu. | Kéo đồng đều tất cả các lớp, không phân biệt vai trò đặc thù của lớp ranh giới (Neutral boundary). |
| **Hermans et al.** [14] | 2017 | Batch-Hard Triplet Loss với hardest positive và hardest negative. | Đột phá trong Person Re-ID và metric learning. | Chưa được điều chỉnh cho lớp ranh giới thiểu số kết hợp phân loại liên hợp với Cross-Entropy. |

---

## 3. Xác Định Khoảng Trống Nghiên Cứu và Phân Định Đóng Góp

### 3.1. Khoảng trống nghiên cứu (Research Gaps)
1. **Bản chất hình học của lớp Neutral bị bỏ qua**: Phần lớn các phương pháp xử lý mất cân bằng đối xử với Neutral như một lớp thiểu số thông thường (như một loài hiếm trong bài toán nhận diện sinh vật). Tuy nhiên, về mặt ngữ nghĩa học (semantics), Neutral không phải là một "cực" riêng biệt mà nằm ngay tại **ranh giới chuyển tiếp** giữa Negative và Positive.
2. **Nghịch lý Cost-Sensitive**: Tăng trọng số phạt (weighted CE) cho Neutral làm tăng Recall nhưng phá hủy Precision (False Positive tăng vọt), dẫn đến suy giảm độ chính xác tổng thể và Macro-F1 không đạt mức tối ưu.
3. **Thiếu cơ chế điều chỉnh ranh giới cục bộ (Boundary-Aware)**: SupCon tiêu chuẩn kéo mọi cụm cùng lớp lại gần nhau, nhưng không ưu tiên tối đa việc tách biệt Neutral khỏi các mẫu phân cực nằm sát ranh giới.

### 3.2. Phần kế thừa từ nghiên cứu trước
- Sử dụng backbone chuẩn `vinai/phobert-base` và tokenizer BPE có tách từ (word segmentation).
- Cấu trúc Cross-Entropy tiêu chuẩn và Class-weighted Cross-Entropy (B1, B2).
- Khung lý thuyết Supervised Contrastive Learning (Khosla et al., 2020) và Batch-Hard Triplet Loss (Hermans et al., 2017).
- Kỹ thuật Post-hoc Logit / Bias Adjustment dựa trên tập Validation (Menon et al., 2021; Provost, 2000).

### 3.3. Giả thuyết đóng góp riêng (Research Hypothesis of NBAC-PhoBERT)
- **Giả thuyết 1 (Feature Space Boundary Separation)**: Thay vì áp dụng contrastive loss đối xứng cho mọi lớp, việc thiết kế một loss định hướng ranh giới chuyên biệt cho Neutral (**Neutral Boundary Loss** $\mathcal{L}_{NB}$) — buộc mỗi anchor Neutral phải cách xa mẫu phân cực khó nhất (hardest negative $q_i \in \{Neg, Pos\}$) một khoảng margin $m$ lớn hơn khoảng cách tới mẫu Neutral khác ($p_i$) — sẽ giúp không gian tiềm ẩn tạo ra một ranh giới bảo vệ rõ ràng xung quanh vùng Neutral.
- **Giả thuyết 2 (Decoupled Representation & Decision Adjustment)**: Kết hợp $\mathcal{L}_{NB}$ (để tạo feature representation tách biệt) với tinh chỉnh bias Neutral $b_N$ trên Validation (để cân bằng Precision-Recall trade-off) sẽ vượt trội hơn việc chỉ dùng weighted Cross-Entropy hoặc chỉ dùng post-hoc logit adjustment trên mô hình baseline.
- **Tính khoa học trung thực**: Đây là giả thuyết nghiên cứu cần được kiểm chứng thực nghiệm có đối chứng (Ablation B4, B5, B6, B7), không mặc định là phương pháp đề xuất sẽ vượt trội trước khi có dữ liệu thực tế.

---

## 4. Tài Liệu Tham Khảo (References)

1. **Nguyen, K. V., Nguyen, V. D., Nguyen, P. X., Truong, T. T., & Nguyen, N. L. T.** (2018). *UIT-VSFC: Vietnamese Students’ Feedback Corpus for Sentiment Analysis*. In Proceedings of the 2018 10th International Conference on Knowledge and Systems Engineering (KSE), pp. 19–24. IEEE.
2. **Nguyen, D. Q., & Nguyen, A. T.** (2020). *PhoBERT: Pre-trained language models for Vietnamese*. In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 1037–1042.
3. **Vu, T., Nguyen, D. Q., Nguyen, D. Q., Dras, M., & Johnson, M.** (2018). *VnCoreNLP: A Vietnamese natural language processing toolkit*. In Proceedings of NAACL: Demonstrations, pp. 56–60.
4. **PyVi**: Python Vietnamese NLP Toolkit. https://github.com/trungtv/pyvi
5. **Dang, T. V., Duong, H. N., & Nguyen, N. L. T.** (2023). *Vietnamese Sentiment Analysis: An Overview and Comparative Study of Fine-tuning Pretrained Language Models*. ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP), 22(6), Article 139.
6. **Nguyen, H. T., & Nguyen, V. C.** (2021). *Deep Learning Approaches for Vietnamese Sentiment Analysis on Student Feedback*. VNU Journal of Science: Computer Science and Communication Engineering.
7. **Hugging Face Hub Benchmarks**: `ura-hcmut/UIT-VSFC`, `tmt3103/VSFC-sentiment-classify-phoBERT`.
8. **Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S.** (2019). *Class-balanced loss based on effective number of samples*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9268–9277.
9. **Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P.** (2017). *Focal loss for dense object detection*. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 2980–2988.
10. **Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., & Kumar, S.** (2021). *Long-tail learning via logit adjustment*. In International Conference on Learning Representations (ICLR).
11. **Ren, J., Yu, C., Sheng, S., Ma, X., Zhao, H., Yi, S., & Li, H.** (2020). *Balanced meta-softmax for long-tailed visual recognition*. Advances in Neural Information Processing Systems (NeurIPS), 33, pp. 4175–4186.
12. **Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., & Krishnan, D.** (2020). *Supervised contrastive learning*. Advances in Neural Information Processing Systems (NeurIPS), 33, pp. 18661–18673.
13. **Chen, T., Kornblith, S., Norouzi, M., & Hinton, G.** (2020). *A simple framework for contrastive learning of visual representations*. In International Conference on Machine Learning (ICML), pp. 1597–1607.
14. **Hermans, A., Beyer, L., & Leibe, B.** (2017). *In defense of the triplet loss for person re-identification*. arXiv preprint arXiv:1703.07737.
15. **Shen, L., Lin, Z., & Huang, Q.** (2016). *Relay backpropagation for effective learning of deep convolutional neural networks*. In European Conference on Computer Vision (ECCV), pp. 467–482.
16. **Kang, B., Xie, S., Rohrbach, M., Yan, Z., Gordo, A., Feng, J., & Kalantidis, Y.** (2020). *Decoupling representation and classifier for long-tailed recognition*. In International Conference on Learning Representations (ICLR).
17. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q.** (2017). *On calibration of modern neural networks*. In International Conference on Machine Learning (ICML), pp. 1321–1330.
18. **Provost, F.** (2000). *Machine learning from imbalanced data sets 101*. In AAAI Workshop on Imbalanced Data Sets, pp. 1–3.
19. **Lipton, Z. C., Elkan, C., & Narayanaswamy, B.** (2014). *Optimal thresholding of classifiers to maximize F1 measure*. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (ECML PKDD), pp. 225–239.
