# Kiểm tra kế hoạch final test

Trạng thái: **PASS — AWAITING_USER_APPROVAL_NOT_EXECUTED**. Đây là kiểm tra kế hoạch và manifest; không fit, predict hay đánh giá test.

- JSON hợp lệ và thỏa JSON Schema draft 2020-12 tại `final_test_plan.schema.json`; đồng thời đã kiểm tra các ràng buộc chéo giữa cấu hình và phép so sánh.
- Có đúng 6 cấu hình: 100% word (2), 100% character (2), 10% word (2); seed 42, 3.166 hàng test dự kiến. Primary duy nhất là A2 − A1; 10% weighting và RQ1 là secondary; character là exploratory.
- 1.142 chỉ số 10% được đọc từ manifest đã lưu, không lấy mẫu lại. Chỉ số không lặp, trong phạm vi train; SHA-256 `4abf51afd48f3af26b29fff46db41406a7050d2f4684a6c01538df6c8b5e4a29` và số lớp 532/46/564 khớp baseline. Các cấu hình full train dùng cùng hash baseline.
- Hash nội dung ba split, tham số TF-IDF và các tham số classifier chung khớp manifest thực nghiệm; hai biến thể class_weight được khai báo rõ. Regex khớp `r"(?u)\b\w+\b"` của pipeline baseline.
- Paired bootstrap chính giữ nguyên 10.000 lần, seed 2026, khoảng 95%; các so sánh khác không được nâng lên primary. Neutral-ratio vẫn chỉ là validation exploratory.
- Script tạo báo cáo không còn ghi đè kế hoạch. Run manifest hiện có ghi `test_evaluated=false`; không tìm thấy test results/predictions trong extended validation.

RQ1 trên test chỉ so sánh hai điểm 10% và 100% với một seed, không thay thế learning curve validation. Lần chạy test cần xác minh lại source hash và train indices trước khi fit.
