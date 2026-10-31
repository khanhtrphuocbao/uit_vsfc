# Nghiên Cứu & Đánh Giá Phân Loại Cảm Xúc Lớp Neutral Trên UIT-VSFC
### *Multi-Paradigm Fused Ensemble & Boundary-Aware Contrastive Learning for Imbalanced Sentiment Analysis*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange)](https://huggingface.co/docs/transformers)
[![Tests](https://img.shields.io/badge/pytest-10%20passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 1. Giới Thiệu & Bối Cảnh (Overview)

Dự án này là công trình nghiên cứu và thực nghiệm chuyên sâu về giải quyết bài toán mất cân bằng lớp cực đoan (imbalanced learning) trong phân tích cảm xúc (Sentiment Analysis) tiếng Việt trên tập dữ liệu phản hồi sinh viên **UIT-VSFC (Vietnamese Students' Feedback Corpus)** — công bố bởi Nguyen et al. (KSE 2018).

### Thách Thức Cốt Lõi: Điểm Nghẽn Lớp Neutral
* **Mất cân bằng dữ liệu cực đoan**: Trong tập Train (11.426 mẫu), lớp **Neutral** chỉ chiếm vỏn vẹn **458 mẫu (~4.01%)**, trong khi hai lớp phân cực Negative và Positive chiếm xấp xỉ 96% (tỷ lệ chênh lệch lên đến `1 : 11.6 : 12.3`).
* **Vùng ranh giới nhạy cảm (Boundary Region)**: Lớp Neutral là vùng đệm cảm xúc trung gian. Các nhận xét khách quan, mô tả nghiệp vụ đào tạo ("*môn học 3 tín chỉ*", "*thầy bắt đầu đúng giờ*") rất dễ bị mô hình gom nhầm sang hai cực phân cực.
* **Nghịch lý Cost-Sensitive (Weighting Paradox)**: Khi ép trọng số phạt để kéo Recall của Neutral lên, số lượng **False Positives (FP) thường bùng nổ dữ dội**, làm sụt giảm nghiêm trọng Macro-Precision và độ chính xác toàn cục (Accuracy).

Dự án phát triển và đối chứng **13 cấu hình thực nghiệm chuẩn hóa** từ baseline tuyến tính đến mô hình **Multi-Paradigm Fused Ensemble (B11)** đạt đỉnh cao hiệu năng SOTA mới trên UIT-VSFC Test Set.

---

## 🏆 2. Bảng Tổng Hợp Kết Quả Thực Nghiệm (Benchmark Results)

Toàn bộ các mô hình học sâu đều được chuẩn hóa huấn luyện đồng nhất **3 epochs** trên GPU, sử dụng cùng bộ tiền xử lý từ tiếng Việt (PyVi), learning rate `2e-5`, và checkpoint chọn lọc duy nhất trên tập Validation (đảm bảo **Zero Data Leakage**).

| Mã Cấu hình | Kiến trúc / Kỹ thuật | Epochs | Macro-F1 | Neutral F1 | Neutral Prec | Neutral Rec | Neutral FP | Neutral FN | Accuracy | Δ Macro-F1 (vs B1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0 (None)** | TF-IDF + Logistic Regression (None) | — | 0.6521 | 0.1311 | 0.7500 | 0.0719 | **4** | 155 | 0.8901 | -0.1690 |
| **B0 (Balanced)** | TF-IDF + Logistic Regression (Balanced) | — | 0.7147 | 0.3459 | 0.2974 | 0.4132 | 163 | 98 | 0.8645 | -0.1064 |
| **B1 (CE)** | PhoBERT-base + Cross-Entropy Tiêu Chuẩn | 3 | 0.8211 | 0.5522 | 0.7327 | 0.4431 | 27 | 93 | 0.9384 | 0.0000 |
| **B2 (Weighted)** | PhoBERT-base + Weighted Cross-Entropy | 3 | 0.8342 | 0.5972 | 0.7107 | 0.5150 | 35 | 81 | 0.9365 | +0.0131 |
| **B3 (CE + Calib)**| PhoBERT-base + CE + Val Logit Bias | 3 | 0.8299 | 0.5772 | 0.6565 | 0.5150 | 45 | 81 | 0.9384 | +0.0088 |
| **B4 (Sampler)** | PhoBERT-base + CE + Class-Aware Sampler | 3 | 0.8339 | 0.5955 | 0.6479 | 0.5509 | 50 | 75 | 0.9356 | +0.0128 |
| **B5 (SupCon)** | PhoBERT-base + SupCon + Sampler | 3 | 0.8353 | 0.5944 | 0.7143 | 0.5090 | 34 | 82 | 0.9394 | +0.0142 |
| **B6 (NBAC)** | PhoBERT-base + NBAC Loss + Sampler | 3 | 0.8251 | 0.5695 | 0.6370 | 0.5150 | 49 | 81 | 0.9346 | +0.0040 |
| **B7 (NBAC+Calib)**| NBAC-PhoBERT + Val Logit Bias | 3 | 0.8286 | 0.5796 | 0.6190 | 0.5449 | 56 | 76 | 0.9346 | +0.0075 |
| **B8 (Bal-SupCon)**| PhoBERT + Smoothed Balanced-SupCon | 3 | 0.8341 | 0.5886 | 0.6667 | 0.5269 | 44 | 79 | 0.9394 | +0.0130 |
| **B9 (Weight-SupCon)**| PhoBERT + Weighted-CE + SupCon + Sampler | 3 | 0.8344 | 0.5918 | 0.6850 | 0.5210 | 40 | 80 | 0.9387 | +0.0133 |
| **B10 (W-SupCon+Cal)**| B9 + Validation Logit Bias | 3 | 0.8355 | 0.5944 | 0.7143 | 0.5090 | 34 | 82 | 0.9397 | +0.0144 |
| **B11 (SOTA Ensemble)**| **Multi-Paradigm Fused Ensemble (B2+B5+B8+B9)**| 3 | **0.8433** | **0.6143** | **0.7143** | **0.5389** | **36** | **77** | **0.9419** | **+0.0222** |

### Kiểm Định Thống Kê Paired Bootstrap (1.000 Lần Lấy Mẫu):
* So sánh **B11 (Fused Ensemble)** với **B1 (PhoBERT-base chuẩn)**:
  - $\Delta \text{Macro-F1} = \mathbf{+0.0222}$, khoảng tin cậy 95%: **[+0.0051, +0.0404]**, $p = 0.010 < 0.05$ (Có ý nghĩa thống kê rõ rệt).
  - $\Delta \text{Neutral-F1} = \mathbf{+0.0621}$, khoảng tin cậy 95%: **[+0.0141, +0.1145]**, $p = 0.009 < 0.01$.

---

## 💡 3. Các Phương Pháp Nổi Bật (Methodology)

```
                                  ┌───────────────────────────────┐
                                  │      UIT-VSFC Feedback        │
                                  └──────────────┬────────────────┘
                                                 │ (PyVi Word Segmentation)
                                                 ▼
                                  ┌───────────────────────────────┐
                                  │   PhoBERT-base (12-layer)     │
                                  └──────────────┬────────────────┘
                                                 │ CLS Representation
                     ┌───────────────────────────┼───────────────────────────┐
                     ▼                           ▼                           ▼
        ┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
        │  Cost-Sensitive (B2)    │ │ Representation (B5, B9) │ │  Prior-Adjustment (B8)  │
        │ Weighted Cross-Entropy  │ │ Supervised Contrastive  │ │ Balanced Softmax Loss   │
        └────────────┬────────────┘ └────────────┬────────────┘ └────────────┬────────────┘
                     │                           │                           │
                     └───────────────────────────┼───────────────────────────┘
                                                 ▼
                                  ┌───────────────────────────────┐
                                  │ Multi-Paradigm Fused Ensemble │
                                  │          (Model B11)          │
                                  │  Macro-F1: 0.8433 | Neu: 0.61 │
                                  └───────────────────────────────┘
```

1. **Neutral-Boundary-Aware Contrastive Loss (NBAC)**: Thiết kế hàm mất mát tương phản có trọng số phân loại và ràng buộc lề động (margin) giữa các cặp (Neutral, Polar).
2. **Balanced-SupCon & Weighted-SupCon**: Khắc phục hiện tượng sụp đổ hình học biểu diễn khi lớp đa số áp đảo bằng cách kết hợp cơ chế lấy mẫu Class-Aware Sampler và cân bằng logit gradient theo phân phối tiên nghiệm.
3. **Multi-Paradigm Fused Ensemble (B11)**: Tổng hợp xác suất theo trọng số tối ưu từ 4 trường phái đại diện: Phạt chi phí (B2), Gom cụm tương phản (B5), Triệt tiêu tiên nghiệm (B8) và Hỗn hợp (B9).

---

## 📂 4. Cấu Trúc Dự Án (Project Structure)

```bash
uit_vsfc/
├── configs/                              # File cấu hình thí nghiệm B0 -> B7 (JSON)
│   ├── B0_tfidf_lr.json
│   ├── B1_ce.json
│   ├── B2_weighted_ce.json
│   ├── B3_ce_val_bias.json
│   ├── B4_ce_sampler.json
│   ├── B5_supcon.json
│   ├── B6_nbac.json
│   └── B7_nbac_val_bias.json
├── data/                                 # Dữ liệu UIT-VSFC gốc
│   ├── train.json                        # 11.426 mẫu huấn luyện
│   ├── valid.json                        # 1.583 mẫu validation
│   └── test.json                         # 3.166 mẫu kiểm thử
├── outputs/                              # Kết quả dự đoán máy đọc được & biểu đồ
│   ├── B0_tfidf_lr/                      # Dự đoán npz & kết quả B0
│   ├── B1_ce/ ... B11_fused_ensemble/    # Checkpoint json & preds.npz từng mô hình
│   └── figures/                          # Confusion matrix & Precision-Recall curves
├── research/                             # Tài liệu học thuật & báo cáo khoa học
│   ├── full_scientific_report.md         # Báo cáo khoa học hoàn chỉnh
│   ├── results_report.md                 # Báo cáo phân tích thực nghiệm chi tiết
│   ├── experiment_plan.md                # Kế hoạch thí nghiệm & giao thức
│   ├── related_work.md                   # Tổng quan các công trình liên quan
│   ├── UIT_VSFC_Scientific_Paper.tex     # Bản thảo bài báo khoa học chuẩn LaTeX
│   └── UIT_VSFC_Scientific_Paper_Report.docx # Báo cáo định dạng Word
├── scripts/                              # Các script thực thi dòng lệnh
│   ├── train.py                          # Huấn luyện mô hình đơn lẻ
│   ├── run_b0_baseline.py                # Chạy baseline TF-IDF + Logistic Regression
│   ├── run_standardized_3epochs.py       # Chạy chuẩn hóa 3 epochs cho B1 - B7
│   ├── run_breakthrough.py               # Chạy nghiên cứu đột phá B8 - B10
│   ├── run_grid_search.py                # Tìm siêu tham số (lambda, margin) trên Val
│   ├── run_experiments.sh                # Shell script chạy chuỗi thực nghiệm tự động
│   ├── eval_report.py                    # Tổng hợp kết quả và tính Paired Bootstrap
│   ├── plot_results.py                   # Vẽ biểu đồ CM và learning curves
│   ├── compute_calibrations.py           # Hiệu chỉnh validation logit bias
│   ├── build_notebook.py                 # Công cụ tự động dựng notebook từ src
│   └── export_docx_report.py             # Xuất báo cáo sang định dạng docx
├── src/                                  # Mã nguồn mô đun hóa (Core Modules)
│   ├── model.py                          # Kiến trúc PhoBERT + Projection Head
│   ├── trainer.py                        # Vòng lặp huấn luyện, đánh giá & early-stop
│   ├── dataset.py                        # Dataset UIT-VSFC & tiền xử lý PyVi
│   ├── losses.py                         # Triển khai CE, Weighted-CE, SupCon, NBAC
│   ├── metrics.py                        # Tính Macro-F1, Precision, Recall, Bootstrap
│   ├── sampler.py                        # Class-Aware Batch Sampler
│   └── calibration.py                    # Thuật toán tìm kiếm bias tối ưu trên Val
├── tests/                                # Bộ kiểm thử đơn vị (Unit Tests)
│   ├── test_losses.py                    # Kiểm thử các hàm loss (CE, SupCon, NBAC)
│   ├── test_metrics.py                   # Kiểm thử độ chính xác các chỉ số F1, CI
│   └── test_sampler.py                   # Kiểm thử phân phối batch của sampler
├── UIT_VSFC_Neutral_Classification.ipynb # Jupyter Notebook trực quan hoá toàn bộ
├── .gitignore                            # Cấu hình bỏ qua file tạm, venv, cache
└── README.md                             # Tài liệu hướng dẫn dự án
```

---

## 🚀 5. Hướng Dẫn Cài Đặt & Sử Dụng (Getting Started)

### 5.1. Khởi Tạo Môi Trường
Yêu cầu Python 3.10 trở lên và GPU với CUDA để đạt tốc độ huấn luyện tối ưu:

```bash
# Clone hoặc điều hướng vào thư mục dự án
cd uit_vsfc

# Tạo và kích hoạt môi trường ảo
python3 -m venv .venv
source .venv/bin/activate

# Cài đặt các gói phụ thuộc cần thiết
pip install torch transformers pyvi scikit-learn numpy pandas matplotlib seaborn pytest
```

### 5.2. Chạy Kiểm Thử Đơn Vị (Unit Tests)
Đảm bảo các mô đun hàm mất mát, bộ lấy mẫu và chỉ số đo lường hoạt động chính xác:

```bash
PYTHONPATH=. pytest tests/
```
*(Kết quả kỳ vọng: `10 passed in ~4s`)*

---

## 🧪 6. Tái Lập Thực Nghiệm (Reproduction Guide)

### 6.1. Chạy Baseline B0 (TF-IDF + Logistic Regression)
```bash
python scripts/run_b0_baseline.py
```

### 6.2. Huấn Luyện Một Mô Hình Cụ Thể
Ví dụ huấn luyện mô hình B1 (PhoBERT-base + Cross-Entropy) hoặc B5 (SupCon):
```bash
# Huấn luyện B1
python scripts/train.py --exp B1_ce --epochs 3 --batch_size 32 --lr 2e-5

# Huấn luyện B5 với Class-Aware Sampler và SupCon
python scripts/train.py --exp B5_supcon --epochs 3 --batch_size 32 --lr 2e-5 --lambda_contrastive 0.1
```

### 6.3. Tái Lập Toàn Bộ Bộ Chuẩn Hóa & Bứt Phá SOTA
Để chạy toàn bộ các mô hình từ B1 đến B11 và xuất kết quả:
```bash
# Chạy bộ B1 - B7 chuẩn hóa 3 epochs
python scripts/run_standardized_3epochs.py

# Chạy bộ B8 - B10 (Balanced & Weighted SupCon)
python scripts/run_breakthrough.py

# Vẽ lại toàn bộ biểu đồ kết quả
python scripts/plot_results.py

# Trích xuất bảng so sánh và kiểm định Bootstrap
python -c "
import pandas as pd
from scripts.eval_report import load_experiment_results, compute_paired_comparisons, generate_markdown_summary
df = load_experiment_results('outputs')
summary = df.groupby('exp_name').agg({'seed': 'count', 'test_macro_f1': ['mean'], 'test_neu_f1': ['mean'], 'test_acc': ['mean']})
print(summary)
"
```

### 6.4. Chạy Jupyter Notebook
Mở notebook hoàn chỉnh để xem biểu đồ trực quan, phân tích mẫu sai (error analysis) và dự đoán trực tiếp:
```bash
jupyter notebook UIT_VSFC_Neutral_Classification.ipynb
```

---

## 📖 7. Trích Dẫn & Tài Liệu Tham Khảo (References)

Nếu bạn sử dụng mã nguồn hoặc kết quả nghiên cứu này, vui lòng trích dẫn các tài liệu liên quan:

1. **UIT-VSFC Corpus**:
   ```bibtex
   @inproceedings{nguyen2018uit,
     title={UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis},
     author={Nguyen, Kiet Van and Nguyen, Vu Duc and Nguyen, Phu X. V. and Truong, Tham T. H. and Nguyen, Ngan Luu-Thuy},
     booktitle={2018 10th International Conference on Knowledge and Systems Engineering (KSE)},
     pages={19--24},
     year={2018},
     organization={IEEE}
   }
   ```
2. **PhoBERT**:
   ```bibtex
   @inproceedings{nguyen2020phobert,
     title={PhoBERT: Pre-trained language models for Vietnamese},
     author={Nguyen, Dat Quoc and Nguyen, Anh Tuan},
     booktitle={Findings of the Association for Computational Linguistics: EMNLP 2020},
     pages={1037--1042},
     year={2020}
   }
   ```

---
*Dự án được thực hiện tại Đại học Công nghệ Thông tin (UIT) — ĐHQG-HCM.*

