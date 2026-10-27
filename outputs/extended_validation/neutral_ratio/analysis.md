# Neutral Ratio — validation

| neutral_count | class_weight | macro_f1_mean | neutral_precision_mean | neutral_recall_mean | neutral_f1_mean | accuracy_mean |
|---|---|---|---|---|---|---|
| 46.0000 | balanced | 0.6775 | 0.4096 | 0.2100 | 0.2750 | 0.8579 |
| 46.0000 | none | 0.5916 | 0.0000 | 0.0000 | 0.0000 | 0.8671 |
| 114.0000 | balanced | 0.6907 | 0.2772 | 0.3927 | 0.3246 | 0.8433 |
| 114.0000 | none | 0.6539 | 0.9153 | 0.1050 | 0.1882 | 0.8690 |
| 228.0000 | balanced | 0.6825 | 0.2259 | 0.5708 | 0.3234 | 0.8187 |
| 228.0000 | none | 0.6955 | 0.3562 | 0.2968 | 0.3235 | 0.8583 |

Chênh lệch Balanced − None ghép theo seed:

| neutral_count | macro_f1_mean | macro_f1_min | macro_f1_max | neutral_f1_mean | neutral_f1_min | neutral_f1_max | accuracy_mean | accuracy_min | accuracy_max |
|---|---|---|---|---|---|---|---|---|---|
| 46.0000 | 0.0859 | 0.0727 | 0.0980 | 0.2750 | 0.2330 | 0.3115 | -0.0093 | -0.0101 | -0.0082 |
| 114.0000 | 0.0368 | 0.0234 | 0.0484 | 0.1364 | 0.0963 | 0.1658 | -0.0257 | -0.0291 | -0.0221 |
| 228.0000 | -0.0131 | -0.0233 | -0.0027 | -0.0001 | -0.0252 | 0.0283 | -0.0396 | -0.0449 | -0.0360 |

Tăng Neutral làm giảm đồng thời Negative và Positive, nên không tách riêng tác động số Neutral. Lựa chọn hàng dùng nhãn sẵn có, không mô phỏng đầy đủ quy trình gán nhãn/active learning. Ba tỷ lệ dùng các phép hoán vị theo lớp lồng nhau. Với 46 Neutral, hash khớp mẫu 10% baseline ở cả ba seed; đã fit lại. Precision/Recall thể hiện đánh đổi, không phải hiệu quả đồng nhất qua mọi tỷ lệ.
