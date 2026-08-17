# Bộ chuẩn bị phỏng vấn — Credit Risk Project

## Pitch 60 giây

Tôi xây dựng một pipeline dự đoán credit default end-to-end trên 30.000 khách hàng. Tôi định nghĩa bài toán theo Probability of Default thay vì chỉ classification 0/1, vì trong risk management cần vừa xếp hạng khách hàng theo mức rủi ro vừa đánh giá chất lượng xác suất. Tôi tạo các behavioral features từ 6 tháng lịch sử chậm trả, bill balance và payment, dùng Logistic Regression làm baseline dễ giải thích và Random Forest làm challenger. Tôi so sánh model bằng ROC-AUC, Average Precision, precision/recall và calibration thay vì accuracy đơn thuần. Sau đó tôi phân tích threshold trade-off và chuyển PD thành các risk band minh họa để business có thể diễn giải portfolio. Tôi cũng ghi rõ giới hạn của dữ liệu và những bước validation/governance cần có trước khi một model tương tự được dùng thật.

## 10 câu interviewer có thể hỏi

### 1. Vì sao chọn Logistic Regression?
Vì nó là baseline mạnh, dễ giải thích, trả probability trực tiếp và giúp kiểm tra xem model phức tạp hơn có tạo incremental value hay không.

### 2. Vì sao không dùng accuracy?
Vì default class mất cân bằng. Một model đoán đa số là non-default vẫn có accuracy cao nhưng có thể bỏ sót phần lớn defaulters.

### 3. ROC-AUC nói điều gì?
Khả năng ranking: một người default có xu hướng được model xếp risk cao hơn một người non-default đến mức nào.

### 4. Recall cao có phải luôn tốt?
Không. Tăng recall thường flag nhiều khách hơn và tạo false positives. Threshold phải dựa trên cost/risk appetite.

### 5. Calibration khác ROC-AUC thế nào?
AUC đo ranking; calibration đo độ đúng của probability tuyệt đối. Model có thể rank tốt nhưng PD bị quá cao hoặc quá thấp.

### 6. Vì sao Random Forest chỉ là challenger?
Phức tạp hơn không đồng nghĩa tốt hơn. Nó phải chứng minh incremental performance đủ lớn để bù lại chi phí explainability/governance.

### 7. Feature nào bạn kỳ vọng quan trọng?
Các biến payment delinquency gần đây, số tháng bị chậm trả, mức utilization và hành vi payment so với bill. Nhưng model importance là association, không phải causality.

### 8. Nếu triển khai thật bạn thay đổi gì?
Dùng out-of-time validation, kiểm tra leakage, stability/drift, fairness, policy/legal eligibility của features, probability calibration, economic thresholding và model monitoring.

### 9. PD, LGD, EAD là gì?
PD = xác suất default; LGD = tỷ lệ tổn thất nếu default; EAD = exposure tại default. Expected Loss thường được diễn giải là PD × LGD × EAD.

### 10. Điểm bạn tự hào nhất trong project?
Không chỉ train model mà đã thiết kế analysis theo tư duy risk: discrimination, detection, calibration, threshold trade-off và governance/limitations.

## Điều tuyệt đối không nói

- “Model này dùng được ngay cho ngân hàng.”
- “Random Forest tốt hơn vì nó phức tạp.”
- “Accuracy cao nghĩa là model tốt.”
- “Feature importance chứng minh nguyên nhân gây default.”
- “Các PD band này là chuẩn ngành.”
