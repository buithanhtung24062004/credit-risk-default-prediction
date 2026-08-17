# Hướng dẫn hiểu project Credit Risk từ A đến Z

## 1. Bài toán thật sự là gì?

Ngân hàng cho khách hàng sử dụng tín dụng. Một số khách hàng trả đúng hạn, một số khách hàng không trả đúng hạn hoặc rơi vào default. Ta muốn dùng thông tin đã quan sát trong quá khứ để ước lượng rủi ro của từng khách hàng trong kỳ tiếp theo.

Target của dataset là:

- `0`: không default tháng sau;
- `1`: default tháng sau.

Nhưng trong risk management, đầu ra có ích hơn chỉ là 0/1 là **Probability of Default (PD)**.

Ví dụ:

- Khách A: PD = 5%
- Khách B: PD = 22%
- Khách C: PD = 61%

Ngay cả khi dùng threshold 50% khiến A và B đều bị dự đoán là "không default", ta vẫn biết B rủi ro cao hơn A rất nhiều. Đây là lý do nên giữ xác suất thay vì chỉ giữ nhãn dự đoán.

---

## 2. Dataset có gì?

UCI dataset có 30.000 quan sát và 23 biến giải thích. Các nhóm biến chính:

### Credit limit

`limit_bal`: hạn mức tín dụng.

### Demographics

`sex`, `education`, `marriage`, `age`.

Trong project học tập ta giữ các biến này để tái hiện benchmark. Trong hệ thống lending thật, việc dùng các thuộc tính nhân khẩu học cần được xem xét kỹ về fairness, luật và governance.

### Payment history

`pay_0`, `pay_2`, ..., `pay_6` thể hiện tình trạng trả nợ trong những tháng gần đây.

Trực giác: khách hàng từng chậm trả nhiều tháng hoặc chậm trả nghiêm trọng thường có tín hiệu rủi ro cao hơn.

### Bill amounts

`bill_amt1` ... `bill_amt6`: số dư hóa đơn trong các tháng trước.

### Previous payments

`pay_amt1` ... `pay_amt6`: số tiền khách đã thanh toán trong các tháng trước.

---

## 3. Tại sao phải EDA trước model?

EDA = Exploratory Data Analysis.

Ta muốn trả lời:

1. Default chiếm bao nhiêu phần trăm?
2. Có missing value không?
3. Có dữ liệu bất thường không?
4. Các biến hành vi của nhóm default khác nhóm non-default như thế nào?
5. Những quan hệ nhìn thấy có hợp lý về mặt tín dụng không?

Điểm quan trọng: EDA không phải cuộc thi vẽ càng nhiều chart càng tốt. Một chart tốt phải giúp trả lời một câu hỏi.

---

## 4. Class imbalance là gì?

Nếu default chỉ chiếm một phần nhỏ dataset thì target bị mất cân bằng.

Giả sử:

- 80% không default;
- 20% default.

Một model ngu ngốc luôn đoán "không default" vẫn có accuracy 80%.

Nhưng nó bắt được 0 khách hàng default, nên gần như vô dụng với risk management.

Vì vậy ta không chọn model chỉ dựa vào accuracy.

---

## 5. Feature engineering trong project

Ta bổ sung các biến có ý nghĩa dễ giải thích.

### `max_delay`

Mức chậm trả tệ nhất trong sáu tháng.

Ví dụ payment status là:

`[-1, 0, 0, 1, 2, 0]`

thì `max_delay = 2`.

### `months_delayed`

Số tháng có tình trạng chậm trả.

Khách chậm một lần và khách liên tục chậm 5/6 tháng rõ ràng không nên được xem giống nhau.

### `avg_bill`

Số dư hóa đơn trung bình.

### `avg_payment`

Khoản thanh toán trung bình.

### `utilization_proxy`

Xấp xỉ:

`average bill / credit limit`

Ví dụ credit limit = 100.000 và average bill = 90.000 thì proxy utilization ≈ 90%.

Mức sử dụng tín dụng cao có thể là tín hiệu hữu ích, nhưng đây chỉ là proxy vì dataset không phải một hệ thống account-level đầy đủ.

### `payment_bill_ratio`

Xấp xỉ số tiền thanh toán trung bình chia cho dư nợ hóa đơn trung bình.

Ta dùng nó như một chỉ báo khả năng/thói quen trả nợ tương đối.

---

## 6. Tại sao chia train/test?

Nếu dùng cùng dữ liệu để vừa huấn luyện vừa đánh giá, model có thể "học thuộc" dữ liệu.

Ta chia:

- 80% train;
- 20% test.

Model chỉ học từ train. Test được giữ riêng để mô phỏng dữ liệu chưa từng thấy.

`stratify=y` giúp tỷ lệ default của train và test gần giống dataset gốc.

---

## 7. Logistic Regression dùng để làm gì?

Logistic Regression là baseline rất phù hợp vì:

1. đơn giản;
2. dễ giải thích;
3. đầu ra là xác suất;
4. có lịch sử sử dụng lâu trong credit scoring;
5. tạo benchmark để biết model phức tạp hơn có thực sự mang lại giá trị không.

Nó mô hình hóa xác suất thông qua log-odds. Bạn không cần nhớ công thức để trình bày project, nhưng nên hiểu ý tưởng: mỗi biến làm tăng hoặc giảm log-odds của default, sau đó được chuyển thành xác suất từ 0 đến 1.

---

## 8. Random Forest dùng để làm gì?

Random Forest gồm nhiều decision tree và tổng hợp kết quả của chúng.

Ưu điểm:

- bắt được quan hệ phi tuyến;
- bắt được tương tác giữa biến;
- ít cần giả định dạng quan hệ hơn logistic regression.

Nhược điểm:

- khó giải thích hơn;
- probability có thể cần calibration;
- một model phức tạp hơn không đồng nghĩa chắc chắn tốt hơn.

Project dùng Random Forest làm **challenger model**, không mặc định coi nó là winner.

---

## 9. ROC-AUC là gì?

Hiểu theo cách trực giác:

Chọn ngẫu nhiên một khách default và một khách non-default. ROC-AUC phản ánh khả năng model xếp khách default lên mức rủi ro cao hơn khách non-default.

- 0.5: gần như random;
- càng gần 1 càng tốt về khả năng ranking.

ROC-AUC không nói xác suất có được calibration đúng hay không.

---

## 10. Precision và Recall

### Recall của default

Trong tất cả khách thực sự default, model bắt được bao nhiêu?

`Recall = TP / (TP + FN)`

False Negative (FN) là khách thực sự default nhưng model bỏ sót.

Trong credit risk, bỏ sót một người rủi ro có thể gây thiệt hại, nên recall thường quan trọng.

### Precision của default

Trong tất cả khách bị model flag là risky/default, bao nhiêu người thực sự default?

`Precision = TP / (TP + FP)`

Nếu ta tăng recall quá mạnh bằng cách flag rất nhiều khách, precision có thể giảm. Đây là trade-off.

---

## 11. Threshold có vai trò gì?

Model trả PD, ví dụ 0.37.

Muốn biến PD thành label 0/1 ta cần threshold.

Nếu threshold = 0.50:

- PD >= 0.50 → predict default;
- PD < 0.50 → predict non-default.

Nếu giảm threshold từ 0.50 xuống 0.30:

- model sẽ flag nhiều khách hơn;
- recall thường tăng;
- false positives cũng thường tăng;
- precision có thể giảm.

Trong doanh nghiệp, threshold nên gắn với cost/risk appetite, không nên mặc định 0.50 một cách máy móc.

---

## 12. Calibration là gì?

Đây là phần rất quan trọng cho PD.

Giả sử một nhóm 1.000 khách có average predicted PD = 20%.

Nếu model calibrated tốt, ta kỳ vọng khoảng 20% nhóm đó thực sự default (trong điều kiện dữ liệu tương tự).

Một model có thể ROC-AUC tốt nhưng calibration kém. Nghĩa là nó xếp hạng đúng ai rủi ro hơn ai, nhưng con số xác suất tuyệt đối bị lệch.

---

## 13. Brier Score là gì?

Brier Score đo sai số bình phương giữa probability dự đoán và outcome 0/1.

Ví dụ khách default thật (`y=1`):

- model A dự đoán 0.90 → sai số nhỏ;
- model B dự đoán 0.10 → sai số lớn.

Brier Score càng thấp càng tốt.

---

## 14. Risk bands để làm gì?

Thay vì đưa cho business một danh sách PD liên tục, có thể chia thành nhóm:

- Low;
- Medium;
- High;
- Very High.

Nó giúp báo cáo portfolio dễ đọc hơn.

Nhưng các cutoff 10%, 25%, 50% trong repo chỉ để minh họa. Một ngân hàng thật cần thiết kế grade/band dựa trên calibration, historical losses, risk appetite, pricing, capital, quy định và chiến lược kinh doanh.

---

## 15. PD, LGD, EAD liên quan thế nào?

Một framework rất phổ biến:

`Expected Loss = PD × LGD × EAD`

### PD — Probability of Default

Khả năng khách hàng default.

### LGD — Loss Given Default

Nếu default xảy ra, ngân hàng mất bao nhiêu phần trăm exposure sau recoveries.

### EAD — Exposure at Default

Tại thời điểm default, ngân hàng đang exposure bao nhiêu tiền.

Project hiện tại tập trung vào PD. Đây là phạm vi hợp lý cho project đầu tiên.

---

## 16. Những điều KHÔNG nên tuyên bố trong CV/phỏng vấn

Không nói:

- "model này có thể dùng ngay cho ngân hàng";
- "Random Forest tốt hơn vì phức tạp hơn";
- "accuracy cao nên model tốt";
- "risk band của tôi là chuẩn ngành";
- "dataset đại diện cho toàn bộ khách hàng hiện tại".

Nên nói:

- đây là portfolio case study;
- mục tiêu là thực hành PD modeling và model evaluation;
- dữ liệu có giới hạn về thời gian và population;
- production validation cần nghiêm ngặt hơn.

---

## 17. Cách kể project trong phỏng vấn trong 60–90 giây

> I built an end-to-end credit-default prediction project using the UCI credit-card dataset with 30,000 customer observations. I framed the problem as estimating Probability of Default rather than only binary classification. I engineered interpretable behavioral features from recent delinquency, bill and payment history, then compared logistic regression as an interpretable baseline with a random forest challenger. I evaluated the models using ROC-AUC, precision-recall metrics and probability calibration rather than accuracy alone because the default class is imbalanced and the business costs are asymmetric. I also converted PD estimates into illustrative portfolio risk bands and documented limitations such as historical sample bias and the need for out-of-time validation in a production setting.

Bạn không cần học thuộc. Hãy hiểu từng ý để có thể diễn đạt bằng lời của mình.

---

## 18. CV bullets sau khi bạn chạy model thật

Sau khi có metrics thực tế, thay `[AUC]`, `[Recall]` bằng số từ `reports/model_metrics.csv`.

- Built an end-to-end credit-default prediction pipeline on 30,000 customer records using Python, feature engineering, logistic regression and Random Forest; achieved test ROC-AUC of **[AUC]**.
- Evaluated default-risk models using ROC-AUC, precision/recall, Brier Score and calibration analysis, emphasizing Probability of Default quality rather than headline accuracy.
- Engineered behavioral credit-risk features from six months of delinquency, billing and payment history and translated predicted PDs into portfolio risk segments for business interpretation.

---

## 19. Thứ tự học project này

Đừng cố đọc toàn bộ code một lần. Học theo thứ tự:

1. hiểu target và PD;
2. hiểu từng nhóm biến;
3. chạy EDA;
4. hiểu feature engineering;
5. hiểu train/test split;
6. chạy Logistic Regression;
7. đọc confusion matrix, recall, precision, ROC-AUC;
8. hiểu calibration;
9. chạy Random Forest để so sánh;
10. tập kể project mà không nhìn notebook.

Nếu bạn làm được 10 bước này, project không còn là code mình làm hộ bạn nữa — nó trở thành project **bạn thực sự hiểu và có thể bảo vệ trong phỏng vấn**.
