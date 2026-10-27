# Báo cáo validation mở rộng UIT-VSFC

## 1. Mục tiêu và câu hỏi

Khảo sát ảnh hưởng khan hiếm dữ liệu, class weighting và thay đổi tỷ lệ Neutral với ngân sách cố định trên cảm xúc phản hồi sinh viên. Macro-F1 là chính; Neutral precision/recall/F1 và accuracy là bổ sung. Dataset chính thức, nguyên văn, ba split cố định; topic chỉ audit. Nguồn được nạp từ cache offline của repository chính thức và hash nội dung khớp audit; revision main không thể xác minh trực tuyến trong phiên này.

## 2. Baseline

Đã kiểm tra 30 cấu hình, summary, confusion, hash, môi trường và mẫu ghép cặp. Ở 100% train, word/None: macro-F1 0.6977, Neutral F1 0.2381, accuracy 0.9097; Balanced: 0.7640, 0.4633, 0.8895. Ba seed ở full train có cùng indices và cùng kết quả, không phải ba phép lặp độc lập. Một câu train trùng văn bản nhưng hai nhãn khác nhau; không sửa hay xóa.

## 3. Neutral Ratio

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


## 4. Phân tích lỗi

# Phân tích lỗi ghép cặp — validation

| transition | negative | neutral | positive |
|---|---|---|---|
| both_correct | 645.0000 | 10.0000 | 719.0000 |
| both_wrong | 28.0000 | 32.0000 | 49.0000 |
| harmed | 29.0000 | 0.0000 | 37.0000 |
| rescued | 3.0000 | 31.0000 | 0.0000 |

Balanced cứu 31 câu Neutral nhưng cả hai vẫn sai 32; 66 câu đúng bởi None trở thành sai bởi Balanced. Các chiều nhầm lẫn chi tiết nằm trong prediction_transitions.csv. Có 48 câu gợi ý đọc thủ công, không phải kết luận nhãn sai. Giá trị predict_proba chưa được kiểm định hiệu chuẩn.


## 5. Ổn định seed

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


## 6. Word so với Character

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


## 7. Kết luận được hỗ trợ

Trên validation này, weighting thường tăng khả năng nhận diện Neutral và macro-F1 ở các mẫu gốc 5% và 10%, với mức tăng biến thiên theo mẫu. Ở 20% Neutral, Balanced có macro-F1 trung bình thấp hơn None: hiệu quả weighting phụ thuộc thành phần dữ liệu. Với full train, weighting tăng Neutral recall từ 10/73 lên 41/73, trong khi accuracy giảm.

## 8. Điều chưa thể kết luận

Chưa có kết quả test, chưa có bằng chứng ý nghĩa thống kê hay hiệu quả tổng quát trên miền dữ liệu khác; chưa chứng minh weighting giải quyết mất cân bằng. Không thể suy ra hiệu ứng độc lập của tăng Neutral khi hai lớp khác bị giảm, hoặc mô phỏng hoàn chỉnh quy trình gán nhãn.

## 9. Giới hạn

Validation chỉ có 73 câu Neutral; các seed dùng chung validation và train subsets có thể chồng nhau. Dữ liệu cùng miền sinh viên; các câu trùng nhãn bất đồng và tính mơ hồ cần người nghiên cứu xem xét. Kết quả lựa chọn trên validation là thăm dò; phân tích mẫu bằng nhãn biết trước. TF-IDF lowercase là hành vi baseline, không phải chỉnh sửa dữ liệu gốc; word token là token bề mặt, không phải phân đoạn từ tiếng Việt.

## 10. Đề xuất final test

So sánh xác nhận chính: full train word, Balanced − None, macro-F1; phụ: Neutral F1/precision/recall và accuracy. Character full train là kiểm tra phụ về biểu diễn. Kế hoạch chi tiết, hash indices và paired bootstrap dự kiến nằm trong final_test_plan.json. **Chưa chạy test; chờ người dùng phê duyệt và khóa kế hoạch.**
