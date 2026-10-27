# Word và character TF-IDF

| train_fraction | feature | class_weight | macro_f1_mean | neutral_f1_mean | accuracy_mean |
|---|---|---|---|---|---|
| 0.1000 | char | balanced | 0.6661 | 0.2507 | 0.8520 |
| 0.1000 | char | none | 0.5869 | 0.0000 | 0.8602 |
| 0.1000 | word | balanced | 0.6775 | 0.2750 | 0.8579 |
| 0.1000 | word | none | 0.5916 | 0.0000 | 0.8671 |
| 1.0000 | char | balanced | 0.7404 | 0.4100 | 0.8749 |
| 1.0000 | char | none | 0.6935 | 0.2381 | 0.9033 |
| 1.0000 | word | balanced | 0.7640 | 0.4633 | 0.8895 |
| 1.0000 | word | none | 0.6977 | 0.2381 | 0.9097 |

10%: ba mẫu train ghép cặp; 100%: một mẫu đầy đủ, một seed, không báo sai số seed. Balanced tăng macro-F1 ở cả hai biểu diễn trong các điều kiện quan sát; kết quả không chứng minh phổ quát cho biểu diễn khác.
