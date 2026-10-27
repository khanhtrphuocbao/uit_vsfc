# Phân tích lỗi ghép cặp — validation

| transition | negative | neutral | positive |
|---|---|---|---|
| both_correct | 645.0000 | 10.0000 | 719.0000 |
| both_wrong | 28.0000 | 32.0000 | 49.0000 |
| harmed | 29.0000 | 0.0000 | 37.0000 |
| rescued | 3.0000 | 31.0000 | 0.0000 |

Balanced cứu 31 câu Neutral nhưng cả hai vẫn sai 32; 66 câu đúng bởi None trở thành sai bởi Balanced. Các chiều nhầm lẫn chi tiết nằm trong prediction_transitions.csv. Có 48 câu gợi ý đọc thủ công, không phải kết luận nhãn sai. Giá trị predict_proba chưa được kiểm định hiệu chuẩn.
