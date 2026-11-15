# Bảng thuật ngữ AIS-NIDS (dành cho cả nhóm)

Tài liệu này giải thích các thuật ngữ xuất hiện trong kế hoạch 8 tuần. Thuật ngữ tiếng Anh được giữ nguyên vì tài liệu, thư viện và code đều dùng tiếng Anh. Phần tiếng Việt chỉ là cách gọi tham khảo, vì nhiều thuật ngữ ML chưa có bản dịch thống nhất.

---

## 1. Dữ liệu và chuẩn bị đầu vào

**Payload byte**
Mỗi byte trong phần dữ liệu (payload) của một gói tin, có giá trị từ 0 đến 255. File CSV có 1500 cột payload cho mỗi gói tin. Gói ngắn hơn 1500 byte được đệm thêm số 0 (zero padding).

**Sample (mẫu)**
Một dòng của file CSV, tức là một gói tin cùng nhãn của nó.

**Label (nhãn) và LabelEncoder**
Label là tên loại lưu lượng (BENIGN, DDoS, PortScan...). Mô hình chỉ làm việc với số, nên `LabelEncoder` (của scikit-learn) đổi mỗi tên thành một số nguyên (0, 1, 2...). Bảng ánh xạ tên <-> số phải được lưu lại để dùng ở mọi bước sau.

**Normalization (chuẩn hóa) bằng cách chia 255**
Đổi mỗi byte từ khoảng 0-255 về khoảng 0.0-1.0. Thứ tự và độ chênh lệch giữa các byte được giữ nguyên, chỉ đổi thang đo. Mục đích là để quá trình huấn luyện ổn định, vì các lớp đầu của mạng được thiết kế cho đầu vào cỡ 1 chứ không phải cỡ hàng trăm.

**uint8**
Kiểu số nguyên không dấu 8 bit (0-255), mỗi giá trị chiếm 1 byte. Dùng kiểu này cho cột payload để file nhỏ và vừa RAM. Chỉ đổi sang `float32` khi đưa từng batch vào mô hình.

**Capped sample (mẫu giới hạn)**
Tập con của file CSV gốc, lấy tối đa 10.000 dòng cho mỗi lớp (tổng 116.726 dòng). Lưu thành `capped_sample.pkl` để cả nhóm dùng chung, không phải đọc lại file 1,41 triệu dòng.

**Train / validation / test**
- **Train:** dữ liệu để mô hình học.
- **Validation:** dữ liệu giữ lại để kiểm tra tiến độ và đưa ra quyết định trong lúc huấn luyện.
- **Test:** dữ liệu chỉ dùng một lần ở cuối để báo cáo kết quả.

**Stratified split (chia phân tầng)**
Chia dữ liệu sao cho tỉ lệ các lớp trong train, validation và test giống nhau. Dùng `stratify=y` và `random_state=42` để kết quả lặp lại được.

**Class imbalance (mất cân bằng lớp)**
Một số lớp có rất nhiều mẫu (10.000) còn lớp khác rất ít (PortScan 830, XSS 3.341). Mô hình thường học tốt các lớp nhiều mẫu hơn.

---

## 2. Kiến trúc mô hình

**Neural network (mạng nơ-ron)**
Mô hình gồm nhiều lớp (layer) nối tiếp nhau. Mỗi lớp biến các con số đầu vào thành các con số mới.

**Unit (đơn vị / nơ-ron)**
Một "máy tính" nhỏ: nhân mỗi số đầu vào với trọng số riêng, cộng lại, rồi cho qua hàm kích hoạt. "Lớp 512 unit" nghĩa là 512 máy tính như vậy chạy song song và cho ra 512 số.

**Weights (trọng số)**
Các số mà quá trình huấn luyện điều chỉnh. "Mô hình đã học" nghĩa là các trọng số đã được chỉnh để dự đoán tốt hơn.

**MLP (Multilayer Perceptron)**
Mạng nơ-ron cơ bản nhất, gồm các lớp nối đầy đủ (mỗi unit nhận tất cả đầu vào). Trong dự án: `1500 -> 512 -> 256 -> số lớp`. Nhanh và đơn giản, nhưng không quan tâm thứ tự các byte.

**ReLU (Rectified Linear Unit)**
Quy tắc áp dụng cho mỗi số đầu ra của một lớp: số âm thành 0, số dương giữ nguyên (−3 thành 0, 2,5 giữ 2,5). Không có nó, nhiều lớp xếp chồng sẽ gộp thành một phép cộng trọng số lớn và mạng chỉ học được quan hệ đường thẳng. ReLU giúp mạng học được các mẫu phức tạp hơn.

**1D-CNN (One-Dimensional Convolutional Neural Network)**
Mạng dùng các bộ lọc trượt dọc theo chuỗi byte có thứ tự để tìm mẫu cục bộ. Nhận ra cùng một mẫu ở bất kỳ vị trí nào, nhưng huấn luyện chậm hơn MLP trên CPU.

**Convolution layer (lớp tích chập)**
Dùng một bộ lọc (filter) ngắn, ví dụ dài 3, trượt dọc chuỗi. Tại mỗi vị trí, nhân 3 byte với trọng số của bộ lọc rồi cộng lại thành một số. Có nhiều bộ lọc (ví dụ 16), mỗi bộ học để tìm một mẫu khác nhau.

**Filter (bộ lọc)**
Danh sách trọng số ngắn dùng trong lớp tích chập. Trọng số của bộ lọc được học trong quá trình huấn luyện.

**Pooling (gộp)**
Thu nhỏ kết quả của lớp tích chập bằng cách chỉ giữ một giá trị trong mỗi cửa sổ nhỏ. Với max pooling cửa sổ 2, dãy 5, 1, 2, 4 thành 5, 4. Giảm lượng dữ liệu cho các lớp sau, và ghi nhận "mẫu xuất hiện quanh đây" mà không cần vị trí chính xác.

**Flatten (làm phẳng)**
Chuyển đầu ra dạng lưới của các lớp tích chập (số bộ lọc x độ dài) thành một danh sách số dài, vì lớp cuối cần đầu vào dạng danh sách. Ví dụ 16 x 375 thành 6.000 số. Không đổi giá trị, chỉ đổi hình dạng.

**Head (lớp đầu ra)**
Lớp cuối cùng, biến bản tóm tắt bên trong của mô hình thành một điểm số cho mỗi lớp. Được tách thành một phần riêng vì bước học tăng dần sẽ thêm một nơ-ron đầu ra vào lớp này.

**Logits**
Các điểm số thô do `head` đưa ra, trước khi qua softmax.

**Softmax**
Biến các điểm số thành xác suất (tổng bằng 1). Xác suất cao nhất là lớp được dự đoán, và cũng chính là con số mà bộ phát hiện "unknown" so sánh với ngưỡng.

**Closed-set classifier (bộ phân loại tập đóng)**
Chỉ nhận ra các loại tấn công đã được huấn luyện. Trong dự án, đây là MLP hoặc 1D-CNN.

**Open-set (tập mở)**
Khả năng nhận ra "cái này không thuộc loại nào tôi biết". Trong dự án làm đơn giản bằng ngưỡng softmax. Bài báo dùng cách phức tạp hơn (K-means chia nhỏ dữ liệu benign cộng với stacking).

---

## 3. Huấn luyện

**Loss (hàm mất mát) và `CrossEntropyLoss`**
Loss đo mức độ sai của mô hình. `CrossEntropyLoss` thấp khi mô hình đưa xác suất cao cho đúng lớp, và cao khi sai hoặc không chắc chắn. Huấn luyện là giảm loss.

**Optimizer và `Adam`**
Optimizer là quy tắc chỉnh trọng số từng bước để giảm loss. `Adam` tự điều chỉnh độ lớn bước nên là lựa chọn mặc định phổ biến.

**Batch (lô)**
Số gói tin được xử lý trước mỗi lần chỉnh trọng số. Batch 256 nghĩa là 256 mẫu mỗi lần.

**Epoch**
Một lượt duyệt qua toàn bộ dữ liệu train. "Khoảng 10 epoch" là mô hình thấy toàn bộ tập train khoảng 10 lần. Thời gian mỗi epoch quyết định bạn có thể chạy thử lại bao nhiêu lần.

**Overfitting (học vẹt)**
Mô hình nhớ dữ liệu train thay vì học quy luật, nên kết quả trên validation và test kém hơn nhiều so với trên train.

---

## 4. Đánh giá

**Accuracy (độ chính xác)**
Tỉ lệ dự đoán đúng. Với lớp mất cân bằng thì con số tổng có thể che giấu các lớp yếu, nên cần xem theo từng lớp.

**Precision, Recall, F1**
- **Precision:** trong số gói được dự đoán là lớp X, bao nhiêu phần trăm thật sự là X.
- **Recall:** trong số gói thật sự thuộc lớp X, bao nhiêu phần trăm được tìm ra.
- **F1:** trung bình điều hòa của hai số trên.

**Macro-F1**
F1 tính riêng cho từng lớp rồi lấy trung bình, mỗi lớp có trọng số như nhau. Lớp lớn không thể che lớp nhỏ. Dùng làm tiêu chí chính khi so sánh MLP và 1D-CNN.

**Confusion matrix (ma trận nhầm lẫn)**
Bảng cho biết lớp nào bị nhầm thành lớp nào (ví dụ GoldenEye bị nhầm thành Slowhttptest).

**`classification_report`**
Hàm của scikit-learn in precision, recall và F1 cho từng lớp. Chạy một lần trên tập test ở cuối.

---

## 5. Hệ thống AIS-NIDS

**NIDS (Network Intrusion Detection System)**
Hệ thống phát hiện xâm nhập mạng, quan sát lưu lượng và cảnh báo khi thấy tấn công.

**Packet-level và flow-level**
Flow-level gộp nhiều gói thành một "luồng" rồi mô tả bằng thống kê. Packet-level nhìn từng gói, kể cả nội dung payload. Bài báo chọn packet-level để thấy các tấn công phụ thuộc payload.

**Unknown (chưa biết)**
Gói tin mà mô hình không chắc thuộc lớp đã biết nào (xác suất cao nhất thấp hơn ngưỡng).

**Threshold (ngưỡng)**
Giá trị `t`. Nếu xác suất cao nhất của mô hình thấp hơn `t` thì gói bị gắn "unknown". Chọn `t` chỉ dựa trên dữ liệu validation của các lớp đã biết.

**Stream rule (quy tắc luồng)**
Tấn công đến thành chuỗi liên tục. Chuỗi gói unknown ngắn hơn 4 bị bỏ vì coi là dữ liệu lỗi. Chuỗi từ 4 gói trở lên được thu thập cho đến khi kết thúc rồi đem đi học.

**Placeholder tag (nhãn tạm)**
Tên tạm như `Unknown-1` cho lớp mới. Con người đặt tên thật sau. Việc đổi tên không ảnh hưởng gì đến những gì mô hình đã học.

**Memory module (mô-đun bộ nhớ)**
Nơi lưu: (1) exemplars (ILD1), (2) VD1, (3) bộ đệm chứa chuỗi unknown đang thu thập.

**Exemplars (mẫu đại diện / ILD1)**
Khoảng 5% mẫu train của mỗi lớp đã biết, được lưu lại và phát lại khi học lớp mới để mô hình không quên các lớp cũ.

**VD1**
Tập nhỏ có nhãn để mô-đun kiểm chứng đo mức quên. Trong bài báo VD1 lấy từ tập test, nên dự án cần quyết định rõ cách làm và ghi lại.

**Incremental learning (học tăng dần)**
Học thêm một lớp mới mà không huấn luyện lại từ đầu. Quy trình: sao chép mô hình, thêm một nơ-ron đầu ra, học lớp mới, khôi phục trọng số cũ (trừ lớp cuối), tinh chỉnh lại bằng exemplars.

**Catastrophic forgetting (quên thảm họa)**
Mô hình học lớp mới tốt nhưng quên các lớp cũ. Exemplars và mô-đun kiểm chứng sinh ra để giảm và phát hiện vấn đề này.

**Forgetting rate (tỉ lệ quên)**
Với mỗi lớp cũ: độ chính xác của mô hình hiện tại trừ độ chính xác của mô hình sau cập nhật. Ngưỡng của bài báo: không quá 10% cho bất kỳ lớp nào.

**Average incremental accuracy (độ chính xác tăng dần trung bình)**
Độ chính xác trên tất cả các lớp đã học bằng mô hình mới nhất, lấy trung bình. Ngưỡng của bài báo: không giảm quá 5% giữa mô hình cũ và mới.

**Validation gate / validation module (cổng kiểm chứng)**
Bước tự động quyết định giữ hay bỏ bản cập nhật dựa trên forgetting rate và average incremental accuracy. Đây là phần làm hệ thống "tự duy trì" (self-sustaining).

**Benign negative test (kiểm tra phủ định bằng dữ liệu benign)**
Đưa dữ liệu benign vào như thể đó là tấn công mới. Kết quả mong đợi: cổng kiểm chứng từ chối cập nhật. Trong bài báo, forgetting trên BENIGN tăng lên 30% và độ chính xác trung bình giảm từ 74% xuống 60%.

---

## 6. Từ viết tắt khác trong dự án

| Từ | Nghĩa |
|---|---|
| CIC-IDS2017 | Bộ dữ liệu tấn công mạng của Đại học New Brunswick, nguồn của file CSV |
| PCAP | File ghi lại gói tin mạng thô |
| CPU-only | Chạy huấn luyện chỉ bằng CPU, không có GPU |
| SENet-I | Bài báo trước đó của nhóm tác giả, giải thích chi tiết CNN đóng và cách tạo ảnh |
| Payload-Byte | Công cụ của tác giả để trích xuất và gắn nhãn gói tin từ PCAP |
