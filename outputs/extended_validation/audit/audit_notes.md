# Kiểm tra baseline

Trạng thái: PASS. Nội dung được nạp từ cache của repository chính thức; hash nội dung khớp cả hai manifest. Không xác nhận được revision main từ chế độ offline.

Hai nhãn khác nhau cho cùng văn bản chuẩn hóa phục vụ audit: xem conflicting_labels.csv. Đây là khả năng mơ hồ hoặc bất nhất, không đủ chứng cứ kết luận nhãn nào sai; giữ nguyên cả hai dòng.

Ở 100%, ba seed có cùng indices; đây không phải ba mẫu train độc lập. {"full_train_unique_sample_hashes": 1, "full_train_metrics_equal_by_weight": {"balanced": true, "none": true}}

Các kiểm tra: {"environment_match": true, "source_hash_schema_counts_match": true, "30_complete_unique_configs": true, "no_test_evaluation": true, "samples_reconstructed_paired": true, "summary_matches": true, "confusions_match_all_metrics": true, "full_train_same_indices": true, "one_conflicting_text": true}
