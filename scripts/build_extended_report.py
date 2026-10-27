from pathlib import Path
import json, hashlib
import pandas as pd,numpy as np
root=Path('outputs/extended_validation'); base=Path('outputs/main_validation')
r=pd.read_csv(root/'neutral_ratio/results.csv'); rs=pd.read_csv(root/'neutral_ratio/summary.csv'); st=pd.read_csv(root/'seed_stability/results.csv'); ss=pd.read_csv(root/'seed_stability/summary.csv'); sp=pd.read_csv(root/'seed_stability/paired_differences.csv'); feat=pd.read_csv(root/'feature_comparison/results.csv'); fs=pd.read_csv(root/'feature_comparison/summary.csv'); pred=pd.read_csv(root/'error_analysis/validation_predictions.csv'); old=pd.read_csv(base/'validation_results.csv'); conf=pd.read_csv(base/'validation_confusions.csv'); samples=json.loads((root/'sample_manifest.json').read_text()); ratio_samples=json.loads((root/'neutral_ratio/sample_manifest.json').read_text())
assert len(r)==18 and len(st)==40 and len(feat)==16 and len(pred)==1583
assert pred.row_index.tolist()==list(range(1583))
assert len(pd.read_csv(root/'error_analysis/review_candidates.csv')) in range(40,61)
for item in samples:
 idx=np.array(item['indices']); assert len(idx)==len(set(idx)) and idx.min()>=0 and idx.max()<11426 and hashlib.sha256(idx.astype('<i8').tobytes()).hexdigest()==item['indices_sha256']
for n in (46,114,228):
 for seed in (42,52,62):
  subset=r[(r.neutral_count==n)&(r.seed==seed)]; s=next(x for x in ratio_samples if x['seed']==seed and x['neutral_count']==n); assert len(subset)==2 and subset.indices_sha256.nunique()==1 and subset.indices_sha256.iloc[0]==s['indices_sha256'] and len(s['indices'])==1142 and s['class_counts']['neutral']==n
for f in (.05,.1):
 for seed in (42,52,62):
  for w in ('none','balanced'):
   a=st[(st.seed==seed)&(st.train_fraction==f)&(st.class_weight==w)].iloc[0]; b=old[(old.seed==seed)&(old.train_fraction==f)&(old.class_weight==w)].iloc[0]
   assert np.isclose(a.macro_f1,b.macro_f1) and np.isclose(a.neutral_f1,b.neutral_f1)
for w in ('none','balanced'):
 for actual in ('negative','neutral','positive'):
  for predicted in ('negative','neutral','positive'):
   count=((pred.true_label==actual)&(pred['pred_'+w]==predicted)).sum(); expected=conf[(conf.seed==42)&(conf.train_fraction==1)&(conf.class_weight==w)&(conf.actual==actual)&(conf.predicted==predicted)]['count'].iloc[0]; assert count==expected
for _,g in feat.groupby(['seed','train_fraction']): assert g.indices_sha256.nunique()==1
assert not list(root.rglob('test_results.csv')) and not list(root.rglob('test_predictions.csv'))
ratio_delta=r.pivot(index=['seed','neutral_count'],columns='class_weight',values=['macro_f1','neutral_f1','accuracy','neutral_precision','neutral_recall']); rd=pd.DataFrame({x:ratio_delta[(x,'balanced')]-ratio_delta[(x,'none')] for x in ['macro_f1','neutral_f1','accuracy','neutral_precision','neutral_recall']}).reset_index(); rd.to_csv(root/'neutral_ratio/paired_differences.csv',index=False)
def table(df,cols):
 d=df[cols].copy()
 for c in d.select_dtypes('number'): d[c]=d[c].map(lambda x: f'{x:.4f}' if pd.notna(x) else '—')
 return '| '+' | '.join(cols)+' |\n|'+'---|'*len(cols)+'\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in d.to_numpy())
cols=['neutral_count','class_weight','macro_f1_mean','neutral_precision_mean','neutral_recall_mean','neutral_f1_mean','accuracy_mean']; rt=table(rs.sort_values(['neutral_count','class_weight']),cols)
stcols=['train_fraction','class_weight','macro_f1_mean','macro_f1_std','neutral_f1_mean','neutral_f1_std','neutral_f1_min','neutral_f1_max','accuracy_mean']; stable=table(ss,stcols)
ft=table(fs,['train_fraction','feature','class_weight','macro_f1_mean','neutral_f1_mean','accuracy_mean'])
transition=pred.groupby(['transition','true_label']).size().unstack(fill_value=0); transition_table=transition.to_markdown() if False else table(transition.reset_index(),['transition','negative','neutral','positive'])
paired=sp.groupby('train_fraction')[['macro_f1','neutral_f1','accuracy']].agg(['mean','std','min','max']); paired.columns=['_'.join(x) for x in paired.columns]; paired_table=table(paired.reset_index(),list(paired.reset_index().columns))
ratio_paired=rd.groupby('neutral_count')[['macro_f1','neutral_f1','accuracy']].agg(['mean','min','max']); ratio_paired.columns=['_'.join(x) for x in ratio_paired.columns]; rpt=table(ratio_paired.reset_index(),list(ratio_paired.reset_index().columns))
full=fs[fs.train_fraction==1]; fullidx=next(x for x in samples if x['phase']=='feature_comparison' and x['train_fraction']==1)
text_ratio=f'''# Neutral Ratio — validation\n\n{rt}\n\nChênh lệch Balanced − None ghép theo seed:\n\n{rpt}\n\nTăng Neutral làm giảm đồng thời Negative và Positive, nên không tách riêng tác động số Neutral. Lựa chọn hàng dùng nhãn sẵn có, không mô phỏng đầy đủ quy trình gán nhãn/active learning. Ba tỷ lệ dùng các phép hoán vị theo lớp lồng nhau. Với 46 Neutral, hash khớp mẫu 10% baseline ở cả ba seed; đã fit lại. Precision/Recall thể hiện đánh đổi, không phải hiệu quả đồng nhất qua mọi tỷ lệ.\n'''
(root/'neutral_ratio/analysis.md').write_text(text_ratio)
text_error=f'''# Phân tích lỗi ghép cặp — validation\n\n{transition_table}\n\nBalanced cứu 31 câu Neutral nhưng cả hai vẫn sai 32; 66 câu đúng bởi None trở thành sai bởi Balanced. Các chiều nhầm lẫn chi tiết nằm trong prediction_transitions.csv. Có {len(pd.read_csv(root/'error_analysis/review_candidates.csv'))} câu gợi ý đọc thủ công, không phải kết luận nhãn sai. Giá trị predict_proba chưa được kiểm định hiệu chuẩn.\n'''
(root/'error_analysis/analysis.md').write_text(text_error)
text_st=f'''# Độ ổn định theo seed\n\nĐộ lệch chuẩn mẫu (ddof=1); tất cả dùng cùng validation, không phải các tập kiểm định độc lập.\n\n{stable}\n\nChênh lệch Balanced − None ghép cùng seed:\n\n{paired_table}\n\nMacro-F1 tăng ở cả 10 seed trong từng fraction, nhưng độ lớn biến thiên; accuracy giảm hoặc không đổi. Đặc biệt Neutral F1 ở 5% Balanced trải từ 0.0482 đến 0.2712.\n'''
(root/'seed_stability/analysis.md').write_text(text_st)
text_feature=f'''# Word và character TF-IDF\n\n{ft}\n\n10%: ba mẫu train ghép cặp; 100%: một mẫu đầy đủ, một seed, không báo sai số seed. Balanced tăng macro-F1 ở cả hai biểu diễn trong các điều kiện quan sát; kết quả không chứng minh phổ quát cho biểu diễn khác.\n'''
(root/'feature_comparison/analysis.md').write_text(text_feature)
report=f'''# Báo cáo validation mở rộng UIT-VSFC\n\n## 1. Mục tiêu và câu hỏi\n\nKhảo sát ảnh hưởng khan hiếm dữ liệu, class weighting và thay đổi tỷ lệ Neutral với ngân sách cố định trên cảm xúc phản hồi sinh viên. Macro-F1 là chính; Neutral precision/recall/F1 và accuracy là bổ sung. Dataset chính thức, nguyên văn, ba split cố định; topic chỉ audit. Nguồn được nạp từ cache offline của repository chính thức và hash nội dung khớp audit; revision main không thể xác minh trực tuyến trong phiên này.\n\n## 2. Baseline\n\nĐã kiểm tra 30 cấu hình, summary, confusion, hash, môi trường và mẫu ghép cặp. Ở 100% train, word/None: macro-F1 0.6977, Neutral F1 0.2381, accuracy 0.9097; Balanced: 0.7640, 0.4633, 0.8895. Ba seed ở full train có cùng indices và cùng kết quả, không phải ba phép lặp độc lập. Một câu train trùng văn bản nhưng hai nhãn khác nhau; không sửa hay xóa.\n\n## 3. Neutral Ratio\n\n{text_ratio}\n\n## 4. Phân tích lỗi\n\n{text_error}\n\n## 5. Ổn định seed\n\n{text_st}\n\n## 6. Word so với Character\n\n{text_feature}\n\n## 7. Kết luận được hỗ trợ\n\nTrên validation này, weighting thường tăng khả năng nhận diện Neutral và macro-F1 ở các mẫu gốc 5% và 10%, với mức tăng biến thiên theo mẫu. Ở 20% Neutral, Balanced có macro-F1 trung bình thấp hơn None: hiệu quả weighting phụ thuộc thành phần dữ liệu. Với full train, weighting tăng Neutral recall từ 10/73 lên 41/73, trong khi accuracy giảm.\n\n## 8. Điều chưa thể kết luận\n\nChưa có kết quả test, chưa có bằng chứng ý nghĩa thống kê hay hiệu quả tổng quát trên miền dữ liệu khác; chưa chứng minh weighting giải quyết mất cân bằng. Không thể suy ra hiệu ứng độc lập của tăng Neutral khi hai lớp khác bị giảm, hoặc mô phỏng hoàn chỉnh quy trình gán nhãn.\n\n## 9. Giới hạn\n\nValidation chỉ có 73 câu Neutral; các seed dùng chung validation và train subsets có thể chồng nhau. Dữ liệu cùng miền sinh viên; các câu trùng nhãn bất đồng và tính mơ hồ cần người nghiên cứu xem xét. Kết quả lựa chọn trên validation là thăm dò; phân tích mẫu bằng nhãn biết trước. TF-IDF lowercase là hành vi baseline, không phải chỉnh sửa dữ liệu gốc; word token là token bề mặt, không phải phân đoạn từ tiếng Việt.\n\n## 10. Đề xuất final test\n\nSo sánh xác nhận chính: full train word, Balanced − None, macro-F1; phụ: Neutral F1/precision/recall và accuracy. Character full train là kiểm tra phụ về biểu diễn. Kế hoạch chi tiết, hash indices và paired bootstrap dự kiến nằm trong final_test_plan.json. **Chưa chạy test; chờ người dùng phê duyệt và khóa kế hoạch.**\n'''
(root/'EXTENDED_VALIDATION_REPORT.md').write_text(report)
print('All output structural, pairing, sample and baseline reproduction checks passed.')
