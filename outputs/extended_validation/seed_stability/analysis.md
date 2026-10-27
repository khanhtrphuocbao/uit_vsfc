# Độ ổn định theo seed

Độ lệch chuẩn mẫu (ddof=1); tất cả dùng cùng validation, không phải các tập kiểm định độc lập.

| train_fraction | class_weight | macro_f1_mean | macro_f1_std | neutral_f1_mean | neutral_f1_std | neutral_f1_min | neutral_f1_max | accuracy_mean |
|---|---|---|---|---|---|---|---|---|
| 0.0500 | balanced | 0.6326 | 0.0210 | 0.1780 | 0.0625 | 0.0482 | 0.2712 | 0.8385 |
| 0.0500 | none | 0.5789 | 0.0056 | 0.0027 | 0.0085 | 0.0000 | 0.0270 | 0.8471 |
| 0.1000 | balanced | 0.6752 | 0.0173 | 0.2726 | 0.0472 | 0.2105 | 0.3423 | 0.8549 |
| 0.1000 | none | 0.5930 | 0.0083 | 0.0107 | 0.0225 | 0.0000 | 0.0533 | 0.8641 |

Chênh lệch Balanced − None ghép cùng seed:

| train_fraction | macro_f1_mean | macro_f1_std | macro_f1_min | macro_f1_max | neutral_f1_mean | neutral_f1_std | neutral_f1_min | neutral_f1_max | accuracy_mean | accuracy_std | accuracy_min | accuracy_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0500 | 0.0537 | 0.0196 | 0.0111 | 0.0811 | 0.1753 | 0.0612 | 0.0482 | 0.2712 | -0.0087 | 0.0036 | -0.0158 | -0.0038 |
| 0.1000 | 0.0822 | 0.0183 | 0.0527 | 0.1133 | 0.2619 | 0.0518 | 0.1648 | 0.3423 | -0.0092 | 0.0059 | -0.0208 | 0.0000 |

Macro-F1 tăng ở cả 10 seed trong từng fraction, nhưng độ lớn biến thiên; accuracy giảm hoặc không đổi. Đặc biệt Neutral F1 ở 5% Balanced trải từ 0.0482 đến 0.2712.
