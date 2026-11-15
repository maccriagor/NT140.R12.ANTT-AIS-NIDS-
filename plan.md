# Nguyên mẫu AIS-NIDS: Kế hoạch 8 tuần

Nhóm 3 người, thời gian 2 tháng, sử dụng laptop cá nhân.
 Sản phẩm bàn giao của kế hoạch này: một **bản demo quay sẵn** của hệ thống có khả năng phân loại các tấn công đã biết, gắn cờ lưu lượng chưa biết, học lưu lượng đó thành một lớp mới , và từ chối các "tấn công" giả ; 1 file báo cáo pdf ; 1 file slide powerpoint.

Bài báo gốc: *AIS-NIDS: An intelligent and self-sustaining network intrusion detection system* (Farrukh và cộng sự). Các số mục bên dưới tham chiếu đến bài báo này.

**Link Google Drive : ** https://drive.google.com/drive/u/0/folders/1pyNg_JhYpXc-qsJ5slYPd23uUgzMe00I
**Link Github : ** https://github.com/maccriagor/NT140.R12.ANTT-AIS-NIDS-
---

## 1. Mục tiêu và phạm vi

**Xây dựng:** một pipeline nhận các dòng gói tin từ file dataset CIC-IDS2017 và:

1. phân loại các tấn công **đã biết** ,
2. gắn cờ các gói tin chưa nhận diện được là **unknown**  ,
3. thu thập một luồng payload mới, gán cho nó một **nhãn tạm** ( ví dụ : `Unknown-1`), và **học nó** như một lớp mới ,
4. **chấp nhận hoặc từ chối** mỗi lần cập nhật bằng một cổng kiểm định (forgetting rate, độ chính xác trung bình),
5. hiển thị quá trình xử lý từ đầu vào đến kết quả đầu ra trên màn hình trong video demo.

**Kế hoạch mở rộng nếu còn thời gian:** bắt gói tin trực tiếp (live sniffing), giai đoạn open-set đầy đủ của bài báo (K-means + stacking), 2D CNN closed-set của bài báo, các tập dữ liệu khác (UNSW-NB15, ACI IoT).

**Lộ trình mô hình:** MLP trước (lựa chọn đầu tiên) -> 1D-CNN (lựa chọn chính nếu lựa chọn đầu tiên thất bại) -> Mô hình CNN của bài báo chỉ như một phần bổ sung tùy chọn.

---

## 2. Thỏa thuận làm việc

### Ba luồng công việc (đổi người phụ trách sau tuần 2)

| Luồng | Phụ trách |
|---|---|
| **A: Dữ liệu và đánh giá** | nạp dữ liệu, chia tập, báo cáo theo từng lớp, bảng forgetting/accuracy, biểu đồ |
| **B: Mô hình và huấn luyện** | MLP, 1D-CNN, đo thời gian mỗi epoch, ngưỡng softmax |
| **C: Pipeline tăng dần và demo** | quy tắc stream, memory module, cập nhật bằng cách thêm một node, cổng kiểm định, kịch bản demo |



### Giao diện dùng chung (thống nhất trong tuần 1, giữ ổn định)

- Đầu vào của mô hình: một batch có shape `(batch, 1500)`, giá trị từ 0 đến 1.
- Đầu ra của mô hình: một batch điểm số các lớp (logits).
- Lớp cuối cùng là một thành phần riêng có tên là `head`.
- Số lớp được lấy từ dữ liệu, không bao giờ gõ tay.
- Các mô hình được tạo bởi một hàm duy nhất (`build_model(name, n_classes)`), nên đổi mô hình chỉ cần sửa một dòng.
- Ánh xạ nhãn (tên <-> số nguyên) được lưu vào một file và dùng lại ở mọi nơi.

### Quy tắc repo

- Code để trong git. File dữ liệu sẽ được đăng lên Google Drive.
- Cố định seed (`random_state=42`) để mọi kết quả có thể so sánh được.
- Một file nhật ký kết quả duy nhất (`results.md`) mà mọi thí nghiệm đều ghi thêm vào: ngày, ai làm, thiết lập, số liệu.


---

## 3. Các thông tin về dữ liệu mà bạn xây dựng dựa trên

- File: CSV payload CIC-IDS2017 đã được xử lý của các tác giả, khoảng 1,41 triệu dòng, 1.505 cột (1500 byte payload, `ttl`, `total_len`, `protocol`, `t_delta`, `label`), được lưu **nhóm theo nhãn**.
- Mẫu giới hạn (capped) : tối đa 10.000 dòng mỗi lớp, **116.726 dòng**, lưu thành `capped_sample.pkl` với các cột payload ở dạng `uint8`.
- **Các lớp chưa biết trong CSV :** DoS Hulk (10.000), DoS slowloris (10.000), Bot (2.543).
- **Các lớp đã biết:** 11 lớp sau khi bỏ Web Attack - Sql Injection.
- Các lớp known nhỏ cần dự đoán là cho kết quả yếu: PortScan (830), Web Attack - XSS (3.341).
- Mẫu lấy các dòng đầu tiên của mỗi lớp theo thứ tự trong file, không phải mẫu ngẫu nhiên. Ghi chú điều này như một hạn chế của đồ án.

---

## 4. Tổng quan

| Tuần | Mục tiêu | Hoàn thành khi |
|---|---|---|
| 1 | Cài đặt, học kiến thức nền, repo chung | Mọi người chạy được loader; số lượng mỗi lớp khớp nhau |
| 2 | Baseline MLP và 1D-CNN song song trên sơ đồ chung | Có báo cáo theo từng lớp và thời gian mỗi epoch cho cả hai |
| **3** | **Gate 1: chọn mô hình. Bộ phát hiện unknown bằng ngưỡng softmax** | Đã đo được tỷ lệ phát hiện unknown |
| **4** | **Quy tắc stream, memory module, cập nhật tăng dần** | Một lớp unknown trở thành lớp known (`Unknown-1`) |
| **5** | **Cổng kiểm định và bài kiểm tra âm tính với benign** | Tấn công thật được chấp nhận, tấn công giả bị từ chối |
| 6 | Ba lớp unknown lần lượt; bảng và biểu đồ theo kiểu bài báo; phần bổ sung tùy chọn | Có số liệu cho cả ba lớp |
| 7 | Kịch bản demo đầu-cuối, bằng chứng về hạn chế, **chốt tính năng** | Chạy trọn vẹn không cần thao tác thủ công |
| 8 | Chạy lại, quay demo cuối, dự phòng | Quay xong |

### Các mốc thời gian quan trọng

1. **Cuối tuần 2/3:** chọn mô hình (quy tắc ở mục 7).
2. **Cuối tuần 4:** nếu cập nhật tăng dần không hoạt động, **dừng mọi việc khác** và sửa nó. Một mô hình yếu hơn nhưng có cập nhật hoạt động tốt hơn một mô hình mạnh hơn nhưng không có.
3. **Đầu tuần 7:** không thêm tính năng mới, chỉ sửa lỗi.

---

## 5. Tuần 1-2

### Tuần 1: cài đặt và kiến thức nền

- Học thêm kiến thức cơ bản về thư viện pandas , kết quả mong đợi từ softmax, vòng lặp huấn luyện PyTorch, lưu/nạp trọng số, độ chính xác theo từng lớp và confusion matrix.
- Đọc: **Algorithm 1**, sau đó **3.2.4 và 3.2.5**, rồi **3.1.3** trong bài báo nghiên cứu.
- **Hoàn thành khi:** cả ba người nạp được mẫu và in ra số lượng mỗi lớp giống hệt nhau.

### Tuần 2: baseline MLP và 1D-CNN

1. **Ngày 1-2 (một người):** viết sơ đồ code chung: nạp dữ liệu, chia tập (phân tầng 70/15/15, seed 42), vòng lặp huấn luyện, hàm đánh giá (báo cáo theo từng lớp kèm thời gian). Cho MLP chạy qua sơ đồ này.
2. **Từ ngày 2 (người thứ hai):** viết 1D-CNN thành một class mô hình riêng, cắm vào sơ đồ.
3. **Người thứ ba:** viết script "softmax gap" và chuẩn bị việc chia mẫu capped thành known/unknown.

MLP: `1500 -> 512 -> 256 -> n_classes`, ReLU giữa các lớp, `head` = lớp cuối.
1D-CNN: hai lớp tích chập kèm pooling, sau đó flatten, rồi `head`.
Huấn luyện cho cả hai: `CrossEntropyLoss`, `Adam`, batch 256, khoảng 10 epoch, in ra loss, độ chính xác trên tập validation và thời gian mỗi epoch.

**Kiểm tra tính hợp lý:** nếu độ chính xác validation nằm gần mức ngẫu nhiên (khoảng 9% với 11 lớp) và không thay đổi, thì có gì đó đang hỏng (đầu vào chưa scale, dữ liệu chưa xáo trộn, nhãn sai).

**Hoàn thành khi:** cả hai mô hình đều có báo cáo theo từng lớp trên tập test, một file trọng số đã lưu, và thời gian mỗi epoch được ghi vào `results.md`.

---

## 6. Chi tiết tuần 3-5

### Tuần 3: chọn mô hình và xây dựng bộ phát hiện unknown

**Mục tiêu**

1. Áp dụng quy tắc quyết định chọn mô hình (mục 7) và ghi lại quyết định cùng các số liệu đằng sau nó.
2. Xây dựng **bộ phát hiện unknown bằng ngưỡng softmax** trên mô hình đã chọn.

**Bộ phát hiện là gì.** Mô hình xuất ra một xác suất cho mỗi lớp known. Lấy xác suất cao nhất . Nếu nó thấp hơn ngưỡng `t`, gói tin bị gắn cờ **unknown**. Nó phát hiện *lưu lượng lạ*, không phải một mẫu cụ thể nào.

**Cách đặt `t`.**
- Chỉ dùng **dữ liệu validation của các lớp known**, ví dụ chọn `t` sao cho 95% gói tin known trong validation nằm trên nó. Không bao giờ tinh chỉnh `t` trên các lớp unknown.
- Thử vài percentile (90, 95, 99) và ghi lại sự đánh đổi.

**Đo (trên dữ liệu test giữ lại)**

| Chỉ số | Ý nghĩa | Tham chiếu bài báo |
|---|---|---|
| Tỷ lệ phát hiện unknown theo lớp | % gói tin Hulk / Bot / slowloris bị gắn cờ unknown | 89% / 90% / 84% |
| Tỷ lệ false-unknown | % gói tin known (đặc biệt là BENIGN) bị gắn cờ sai | không báo cáo |
| Khoảng cách độ tin cậy (confidence gap) | softmax cao nhất trung bình trên gói tin known so với unknown | không báo cáo |

**Công việc**

- [ ] Viết `detect_unknown(model, x, t)` trả về độ tin cậy và một cờ.
- [ ] Hiệu chỉnh `t` trên dữ liệu validation known.
- [ ] Đánh giá trên dữ liệu test known và trên ba lớp unknown.
- [ ] Vẽ histogram độ tin cậy: known so với unknown (Nếu được).
- [ ] Ghi kết quả và `t` đã chọn vào `results.md`.

**Rủi ro**

- Mô hình có thể *đưa ra kết quả sai* trên đầu vào lạ. Nếu khoảng cách nhỏ, hãy thử ngưỡng thấp hơn, một mô hình khác, hoặc temperature scaling trước khi thay đổi bất cứ thứ gì khác.
- Lưu ý: bài báo chỉ gửi những gói tin mà bộ phân loại closed-set gọi là benign đến một mô hình open-set riêng. Phiên bản một-ngưỡng của nhóm là mô hình đơn giản hơn. 

**Hoàn thành khi:** bạn có tỷ lệ phát hiện cho cả ba lớp unknown tại một `t` đã chọn, cộng với tỷ lệ false-unknown.

---

### Tuần 4: quy tắc stream, memory module, cập nhật tăng dần

**Mục tiêu:** đưa **một** lớp unknown (bắt đầu với DoS Hulk) từ "unknown" thành "known" với một nhãn tạm. Đồ án phụ thuộc vào điều này.

#### 4.1 Quy tắc stream

- Một tấn công thật đến dưới dạng một **chuỗi liên tục** các gói tin unknown. Một chuỗi ngắn hơn **4 gói** bị loại bỏ như dữ liệu xấu. Một chuỗi từ 4 gói trở lên được thu thập cho đến khi nó kết thúc, sau đó gửi đi học.
- Cách triển khai: một bộ đếm được đặt lại về 0 mỗi khi một gói tin *không* bị gắn cờ unknown. Giữ ngưỡng 4 làm tham số (bài báo sử dụng con số này để minh họa).
- **Cố ý xây dựng stream cho demo.**Hãy dựng một stream như: khối BENIGN, rồi khối Hulk, rồi lại BENIGN. Nếu bạn xáo trộn tất cả, sẽ không có chuỗi nào hình thành.

#### 4.2 Memory module

| Thành phần | Nội dung | Mục đích |
|---|---|---|
| ILD1 (exemplars) | 5% mẫu huấn luyện của mỗi lớp known | phát lại trong lúc fine-tuning để ngăn quên |
| VD1 | một tập nhỏ có nhãn | dùng bởi bộ kiểm định ở tuần 5 |
| Unknown stream buffer | các gói tin unknown đã thu thập | dữ liệu huấn luyện cho lớp mới |

Đánh đổi: bộ nhớ lớn hơn nghĩa là ít quên hơn nhưng tốn lưu trữ hơn và cập nhật chậm hơn. 5% là giá trị của bài báo và có thể điều chỉnh.

**Quyết định về VD1 (hãy ghi lại).** Trong bài báo, VD1 là 10% của tập con *test* (CS-D3), trùng với dữ liệu dùng để báo cáo. Hãy quyết định có chủ đích: hoặc lấy VD1 từ tập chia **validation** (sạch hơn), hoặc làm theo bài báo và liệt kê sự trùng lặp này như một hạn chế.

**Cách chia lớp unknown (đề xuất).** Hulk có 10.000 dòng (bài báo dùng 5.000 trong thí nghiệm của họ). Dùng khoảng 60% làm stream đến và giữ khoảng 40% không đụng đến cho bài kiểm tra trước/sau. Bot chỉ có 2.543 dòng, nên hãy giảm quy mô tương ứng.

#### 4.3 Cập nhật tăng dần (bài báo 3.2.4)

Công thức: sao chép mô hình (`theta_b` -> `theta_u`), thêm một node đầu ra, sau đó:

1. **Khởi tạo:** `theta_u` nhận tất cả trọng số của `theta_b` trừ node đầu ra mới.
2. **Học lớp mới:** một bước học ban đầu chỉ dùng các mẫu của lớp mới. Việc này làm nhiễu loạn trọng số rất nhiều.
3. **Khôi phục trọng số:** khôi phục tất cả trọng số về giá trị ban đầu **trừ lớp cuối cùng**, lớp này giữ các cập nhật từ bước 2.
4. **Tinh chỉnh lặp:** nhiều vòng huấn luyện trên stream cộng với các exemplar đã lưu, để lấy lại hiệu năng cũ trong khi vẫn giữ lớp mới.

Mở rộng head (bản phác):

```python
import torch, torch.nn as nn

def extend_head(old_head: nn.Linear) -> nn.Linear:
    new = nn.Linear(old_head.in_features, old_head.out_features + 1)
    with torch.no_grad():
        new.weight[:-1] = old_head.weight   # copy old rows
        new.bias[:-1] = old_head.bias       # copy old biases
    return new                               # last row = new class, randomly initialized
```

Đặt tên tạm: gán số lớp trống kế tiếp và tên `Unknown-1`, `Unknown-2`, ... rồi thêm vào ánh xạ nhãn đã lưu. Con người có thể đổi tên sau; việc này không thay đổi gì những điều mô hình đã học.

**Công việc**

- [ ] Bộ dựng stream (benign -> Hulk -> benign).
- [ ] Bộ đếm quy tắc stream kèm unit test (chuỗi 3 bị loại bỏ, chuỗi 4 được giữ).
- [ ] Memory module: lấy mẫu exemplar (5% mỗi lớp), VD1, buffer.
- [ ] `extend_head`, rồi hàm cập nhật 4 bước.
- [ ] Kiểm tra trước/sau trên các dòng Hulk giữ lại.

**Kết quả kỳ vọng (giả thuyết, không phải đảm bảo):** trước khi cập nhật, hầu hết gói tin Hulk giữ lại bị gắn cờ unknown; sau khi cập nhật, hầu hết được gán nhãn `Unknown-1`, và độ chính xác của các lớp cũ chỉ giảm một chút. Số liệu chính xác sẽ khác bài báo (bài báo: khoảng 88% trên lớp mới sau khi học).

**Rủi ro**

- Việc cập nhật làm hại các lớp cũ rất nhiều. Hãy kiểm tra việc lấy mẫu exemplar, learning rate trong lúc tinh chỉnh (dùng giá trị nhỏ), và số vòng tinh chỉnh.
- Lớp mới trông giống một lớp đã có (bài báo thấy điều này với GoldenEye và Slowhttptest bên cạnh Hulk). Hãy dự kiến có một ít sự quên ở đó.

**Gate 2 (cuối tuần 4):** nếu cập nhật không hoạt động, dừng các việc khác và sửa nó.

---

### Tuần 5: cổng kiểm định và bài kiểm tra âm tính với benign

**Mục tiêu:** làm cho hệ thống tự quyết định có giữ một bản cập nhật hay không. Chứng minh nó có thể từ chối việc học giả.

#### 5.1 Các chỉ số (bài báo 3.2.5)

- **Forgetting rate (theo từng lớp cũ):** độ chính xác của mô hình **hiện tại** trừ đi độ chính xác của mô hình **đã cập nhật** trên lớp đó, đo trên VD1.
- **Độ chính xác tăng dần trung bình (average incremental accuracy):** độ chính xác trên tất cả các lớp đã học dùng mô hình mới nhất, lấy trung bình.

#### 5.2 Quy tắc của cổng

Các ngưỡng của bài báo:
- forgetting rate không được vượt quá **10% với bất kỳ lớp riêng lẻ nào**;
- độ chính xác trung bình không được giảm quá **5%** giữa mô hình hiện tại và mô hình đã cập nhật.

Dòng 22 của Algorithm 1 viết điều kiện độ chính xác là "Average Incremental Accuracy < 5%", điều này chỉ hợp lý nếu hiểu là **giảm tối đa 5 điểm**. Hãy dùng cách hiểu đó và ghi chú sự mơ hồ này.

**Chấp nhận** -> giữ mô hình đã cập nhật, lưu các exemplar của lớp mới vào bộ nhớ.
**Từ chối** -> bỏ mô hình đã cập nhật và các mẫu đã thu thập.

**Lựa chọn ngưỡng (hãy quyết định và ghi lại).** Bài báo đặt các ngưỡng sau lớp unknown đầu tiên (Hulk) và dùng lại cho các lớp còn lại. Nghĩa là các ngưỡng được tinh chỉnh trên một trong các trường hợp test. Hoặc làm theo (và liệt kê đó là một hạn chế) hoặc chọn các ngưỡng từ trước. Chính bài báo nói rằng việc chọn ngưỡng chưa được khám phá đầy đủ.

#### 5.3 Bài kiểm tra âm tính với benign (bài báo Table 5)

1. Lấy các mẫu **benign** ngẫu nhiên và đưa vào như thể chúng là một stream tấn công unknown.
2. Chạy cùng cập nhật tăng dần và cùng cổng đó.
3. **Kỳ vọng:** cổng **từ chối** bản cập nhật. Trong bài báo, forgetting trên BENIGN tăng vọt lên 30% và độ chính xác trung bình giảm từ 74% xuống 60%.

**Công việc**

- [ ] `evaluate_forgetting(model_old, model_new, vd1)` trả về số liệu theo từng lớp.
- [ ] `evaluate_avg_accuracy(model, vd1)`.
- [ ] `gate(model_old, model_new, vd1)` trả về accept/reject kèm các số liệu.
- [ ] Chạy bản cập nhật Hulk thật qua cổng: kỳ vọng **accept**.
- [ ] Chạy bài kiểm tra âm tính benign: kỳ vọng **reject**.
- [ ] Ghi cả hai lần chạy vào `results.md`.

**Tiêu chí chấp nhận của tuần**

| Kiểm tra | Kỳ vọng |
|---|---|
| Lớp mới thật (Hulk) | Được chấp nhận |
| Stream giả (benign như unknown) | Bị từ chối |
| Stream ngắn hơn 4 gói tin | Bị loại bỏ trước khi học |

**Rủi ro**

- Cổng chấp nhận stream giả (ngưỡng quá lỏng) hoặc từ chối stream thật (quá chặt). Hãy xem các số forgetting thực tế trước, rồi điều chỉnh, và ghi lại bạn đã đổi gì và vì sao.
- VD1 nhỏ đối với các lớp hiếm (PortScan, XSS), nên số forgetting của chúng bị nhiễu. Hãy báo cáo chúng kèm lưu ý đó.

**Hoàn thành khi:** cả ba hàng tiêu chí chấp nhận đều hoạt động như kỳ vọng và nhóm có thể giải thích, theo từng lớp, vì sao cổng quyết định như vậy.

---

## 7. Chọn mô hình: quy tắc và các tiêu chuẩn (tuần 2-3)

Giữ mọi thứ khác như nhau (cùng cách chia, seed, số epoch, batch size, optimizer). **Thống nhất quy tắc trước khi thấy kết quả.**

| Tiêu chuẩn | Vì sao |
|---|---|
| Macro-F1 trên tập test | lấy trung bình đều giữa các lớp, nên các lớp lớn không thể che giấu các lớp yếu |
| Recall trên các lớp yếu  | nơi CNN có thể thực sự giúp ích |
| Thời gian mỗi epoch và tổng thời gian huấn luyện | quyết định bạn có thể chạy lại bao nhiêu lần |
| Khoảng cách phát hiện unknown | softmax cao nhất trung bình trên gói tin test known so với trên ba lớp unknown |

**Quy tắc đề xuất (cùng nhau điều chỉnh các con số):** chỉ chọn 1D-CNN nếu macro-F1 của nó cao hơn rõ rệt (ví dụ 3+ điểm) hoặc nó sửa được các lớp yếu, **và** thời gian mỗi epoch của nó dưới khoảng 3 lần của MLP. Nếu không thì giữ MLP. Nếu hai mô hình gần nhau, dùng một lần cập nhật tăng dần dùng-một-lần cho mỗi mô hình và so sánh forgetting để phân thắng bại.

**Benchmark trước (30 phút):** nạp mẫu, đo thời gian một epoch của mỗi mô hình, kiểm tra mức dùng RAM, và theo dõi hiện tượng CPU bị throttling sau vài phút.

**Các lớp yếu là điều dự kiến.** Chính bài báo đã báo cáo 66-69% trên GoldenEye, Slowhttptest, PortScan và SQL injection. Số liệu thấp trên các lớp nhỏ hoặc giống nhau có thể đến từ dữ liệu, không phải từ mô hình.

---

## 8. Tuần 6-8

### Tuần 6: cả ba lớp unknown, bảng biểu, phần bổ sung tùy chọn

- Chạy các lớp unknown lần lượt: **Hulk -> Bot -> slowloris**, mỗi lớp đi qua detect, stream, update, gate.
- Tạo các bảng như Table 2-4 của bài báo (độ chính xác theo lớp trước/sau, forgetting rate) và một biểu đồ như Fig. 7 (độ chính xác tăng dần trung bình khi các lớp được thêm vào). Dự kiến có sự suy giảm dần .
- Kiểm tra **độ thuần của nhãn tạm (placeholder purity):** sau đó, xác minh rằng các gói tin được gửi vào `Unknown-1` gần như toàn là các dòng Hulk thật.
- **Tùy chọn :** thêm các cột bổ sung (`ttl`, `total_len`, `protocol`, `t_delta`), mỗi lần một thay đổi, và so sánh các báo cáo theo lớp. Scale: `ttl` / 255, `total_len` / 1500, `protocol` thành các cột 0/1 riêng, `t_delta` với phép biến đổi log. Cẩn thận hiện tượng học lối tắt (shortcut learning).
- **Tùy chọn:** một bản tóm tắt ngắn dễ đọc cho mỗi stream unknown (độ dài gói tin trung bình, tỷ lệ giao thức, thời gian) làm "mô tả mẫu" cho demo.

### Tuần 7: demo đầu-cuối và đóng băng

- Một script duy nhất stream thư mục dữ liệu theo thứ tự và in ra từng giai đoạn: phân loại, cờ unknown, stream đã thu thập, cập nhật, kết quả của cổng, trước/sau.
- Chạy nó từ đầu đến cuối mà không cần thao tác thủ công.
- Thu thập bằng chứng về hạn chế dựa trên kết quả test.


### Tuần 8: quay và dự phòng

- Chạy lại mọi thứ từ trạng thái sạch, quay demo cuối, giữ bản ghi và câu lệnh chính xác đã dùng.
- Dự phòng cho việc sửa lỗi. Không thêm tính năng mới.

---

## 9. Phần bổ sung tùy chọn (chỉ khi mọi thứ xong sớm)

- CNN closed-set của bài báo trên ảnh 128x64x3. **Trước hết hãy làm rõ cách bố trí gói tin trong SENet-I**, và đo thời gian mỗi epoch trước khi cam kết.
- Một tập dữ liệu thứ hai (cần công cụ Payload-Byte và các PCAP lớn; nhiều khả năng quá nặng so với thời gian).
- Phân tích một PCAP ngắn bằng Scapy thành cùng các cột để cho thấy các gói tin thô chảy qua pipeline.