# Tài liệu Phân tích Kỹ thuật Mã nguồn (Codebase Walkthrough)

Tài liệu này phân tích chi tiết cấu trúc thuật toán và luồng xử lý dữ liệu trong tệp mã nguồn `notebooks/DeTai3_Kaggle.ipynb` (áp dụng cho Cell 1 đến Cell 10). Mục tiêu là làm rõ cơ chế luân chuyển dữ liệu, không gian siêu tham số, quá trình lan truyền ngược (backpropagation) và chu trình cập nhật trọng số của mô hình.

---

## Kiến trúc Tổng quan (System Pipeline)

CIFAR-10
   ↓ transform + split cố định
DataLoader
   ↓ mini-batch (images, labels)
ResNet-18
   ↓ logits 10 lớp
CrossEntropyLoss
   ↓ backward tạo gradient
Optimizer cập nhật weights
   ↓ lặp theo epoch
Validation chọn best checkpoint → test đúng một lần → CSV → mean ± std → biểu đồ

---

## Cell 1 - Khởi tạo Môi trường và Cấu hình Thực thi

Cell 1 thiết lập các biến toàn cục điều khiển luồng thực thi và kiểm tra ràng buộc tài nguyên phần cứng:

* `MEMBER`: Định danh phân công tập thực nghiệm cho từng thành viên (không can thiệp vào thuật toán huấn luyện).
* `RUN_ALL_CONFIGS = True`: Kích hoạt chế độ tự động thực thi toàn bộ 26 cấu hình x 2 seeds.
* `DEBUG = True`: Chế độ kiểm thử nhanh (giảm số lượng mẫu dữ liệu, số lượng epoch và seed). Ở chế độ này, chu trình đánh giá trên tập test chủ động bị vô hiệu hóa nhằm tránh lãng phí chi phí tính toán trong giai đoạn tinh chỉnh code.
* `PART`: Định danh phân đoạn dữ liệu nhằm phân tách tệp nhật ký đầu ra giữa các phiên làm việc độc lập.
* `RUN_IDS`: Bộ lọc danh mục thực nghiệm cần thực thi.
* `NOTEBOOK_VERSION`: Mã định danh phiên bản thuật toán lõi phục vụ đồng bộ nội bộ.
* `SAVE_CHECKPOINTS = False`: Cờ tối ưu hóa dung lượng bộ nhớ. Trọng số chỉ lưu tạm thời trên RAM để đánh giá trực tiếp và không lưu trữ 52 tập tệp trọng số lớn xuống đĩa cứng.

**Kiểm soát tài nguyên hạ tầng:**
Hàm `torch.cuda.is_available()` kiểm tra sự tồn tại của thiết bị gia tốc phần cứng. Hệ thống chủ động từ chối dòng GPU Tesla P100 do giới hạn tương thích với phiên bản PyTorch hiện hành và chỉ định thực thi cố định trên `cuda:0` của hạ tầng Kaggle T4 x2. 

Trường hợp dữ liệu đầu vào chứa tập tin nén `cifar-10-python.tar.gz`, thư viện `tarfile` sẽ giải nén tự động vào thư mục có quyền ghi `/kaggle/working`. Biến `DATA_ROOT` được gán vào thư mục cha của `cifar-10-batches-py` nhằm đáp ứng đầu vào tiêu chuẩn của `torchvision.datasets.CIFAR10`.

---

## Cell 2 - Thiết lập Không gian Siêu tham số (Hyperparameter Configuration)

Đối tượng `BASE_CONFIG` đóng vai trò là không gian tham số chuẩn (baseline). Mỗi thực nghiệm đơn lẻ được khởi tạo bằng phương pháp hợp nhất dictionary:

```python
config = {**BASE_CONFIG, **experiment_config, "epochs": EPOCHS}

```

Cơ chế đè (override) từ trái sang phải của Python đảm bảo các thuộc tính định nghĩa trong `experiment_config` sẽ thay thế giá trị tương ứng trong `BASE_CONFIG`. Ví dụ:

```python
experiment("opt_adam", "Adam", "optimizer", optimizer="adam", lr=0.001)

```

Cấu hình trên chỉ cập nhật thuật toán tối ưu và tốc độ học (LR) tương ứng, trong khi giữ nguyên các biến số khác (Batch size, Normalization, Schedule, Epochs). Do LR được điều chỉnh đồng thời, bản chất của phép thực nghiệm này là *"So sánh hiệu năng giữa các thuật toán tối ưu dưới điều kiện tốc độ học được tinh chỉnh tương đương"*, không phải tác động đơn thuần của thuật toán.

`ANCHOR` đóng vai trò là điểm cơ sở đối chứng (baseline reference) dùng chung cho tất cả các nhánh phân tích. Mô hình Anchor chỉ thực thi huấn luyện 01 lần và kết quả được tái sử dụng làm mốc so sánh xuyên suốt 04 nhánh thực nghiệm.

---

## Cell 3 - Quản lý Dữ liệu và Phân chia Tập dữ liệu (Data Pipeline & Splitting)

Hệ thống khởi tạo 03 đối tượng dataset cho tập huấn luyện/kiểm định và 01 đối tượng cho tập kiểm thử từ CIFAR-10:

* `TRAIN_CLEAN`: Dữ liệu huấn luyện không áp dụng kỹ thuật tăng cường (non-augmented).
* `TRAIN_AUGMENTED`: Dữ liệu huấn luyện áp dụng các phép biến đổi ngẫu nhiên (Crop, Horizontal Flip, Color Jitter).
* `EVAL_DATA`: Dữ liệu kiểm định chuẩn hóa (clean transform).
* `TEST_DATA`: Tập dữ liệu kiểm thử độc lập gồm 10,000 mẫu, áp dụng phép biến đổi chuẩn hóa cố định.

**Phương pháp phân chia (Data Splitting Protocol):**
Sử dụng hàm `torch.randperm(..., generator=split_generator)` với `seed = 4653` để tạo hoán vị ngẫu nhiên cố định: 45,000 chỉ số đầu thuộc tập huấn luyện (Training Set) và 5,000 chỉ số sau thuộc tập kiểm định (Validation Set). Quá trình phân chia này hoàn toàn độc lập với `training_seed` (42 và 2026).

`DataLoader` đảm nhận nhiệm vụ gom nhóm dữ liệu thành các lô nhỏ (mini-batches). Tập train sử dụng `shuffle=True` và khởi tạo ngẫu nhiên theo seed để đảm bảo tính tái lập thứ tự dữ liệu. Tập validation/test cố định `batch_size = 256` và `shuffle=False`. Khi mô hình chuyển sang trạng thái đánh giá (`model.eval()`), các lớp BatchNorm sử dụng thống kê tích lũy (running statistics) và các lớp Dropout bị ngắt, do đó kích thước lô đánh giá không làm sai lệch đặc tính huấn luyện của các lô nhỏ (8, 32, 128).

---

## Cell 4 - Kiến trúc Mô hình ResNet-18 Tối ưu cho CIFAR-10

Khối cơ bản (`BasicBlock`) bao gồm 02 lớp tích tụ (convolution) 3 x 3. Luồng tín hiệu đường tắt (residual connection) được hợp nhất với đường chính qua phép cộng element-wise:

```python
return self.relu(outputs + residual)

```

Đường tắt giúp dòng giá trị gradient lan truyền trực tiếp qua các lớp sâu mà không bị suy giảm (vanishing gradient). Trường hợp có sự thay đổi về kích thước không gian hoặc số kênh tín hiệu, một phép tích tụ 1 x 1 tại khối `shortcut` sẽ biến đổi tensor `residual` về cùng kích thước trước khi thực hiện phép cộng.

Khác với kiến trúc ResNet-18 nguyên bản dành cho ImageNet 224 x 224, phiên bản điều chỉnh cho CIFAR-10 32 x 32 thay thế lớp tích tụ đầu tiên bằng kernel 3 x 3, `stride = 1` và loại bỏ lớp Max-pooling ban đầu nhằm bảo toàn độ phân giải không gian của đặc trưng.

**Các phương pháp Chuẩn hóa (Normalization Techniques):**

* **Batch Normalization (BN):** Tính toán trung bình và độ lệch chuẩn theo chiều dọc của lô dữ liệu (N), duy trì giá trị thống kê tích lũy `running_mean` và `running_var`.
* **Layer Normalization 2D (LN):** Chuyển đổi định dạng tensor từ NCHW sang NHWC để áp dụng `nn.LayerNorm(C)` độc lập trên từng vị trí không gian H x W.
* **Group Normalization (GN):** Chia C kênh tín hiệu thành 8 nhóm độc lập để tính toán thống kê, hoàn toàn không phụ thuộc vào kích thước lô dữ liệu N.

Lớp Dropout được bố trí phía sau lớp Global Average Pooling và ngay trước lớp phân loại tuyến tính (Linear Classifier). Trong giai đoạn huấn luyện, Dropout triệt tiêu ngẫu nhiên một tỷ lệ trọng số; khi chuyển sang `model.eval()`, lớp này tự động ngắt hoạt động.

---

## Cell 5 - Thuật toán Tối ưu và Chiến lược Điều chỉnh Tốc độ Học

Hàm `build_optimizer` phân tách các tham số của mô hình thành 02 nhóm chính dựa trên số chiều tensor nhằm áp dụng suy giảm trọng số (weight decay) chính xác:

1. **Nhóm nhận Weight Decay:** Các ma trận trọng số của lớp Convolution và Linear (`ndim > 1`).
2. **Nhóm không nhận Weight Decay:** Các tham số chệch (bias) và tham số học được của các lớp Normalization (`ndim == 1`).

Hàm `learning_rate_for_epoch` tính toán giá trị tốc độ học (LR) theo từng epoch dựa trên các chiến lược (Schedules):

* **Constant:** Duy trì LR không đổi.
* **Step Decay:** Nhẩm tỷ lệ `gamma` sau mỗi khoảng `step_size` epochs.
* **Cosine Annealing:** Giảm LR theo nửa chu kỳ đường mòn Cosine.
* **Warm-up:** Tăng tuyến tính LR từ giá trị khởi tạo nhỏ đến LR cơ sở trong các epoch đầu.

Hàm `set_learning_rate` thực hiện cập nhật thông số LR trực tiếp vào thuộc tính `param_groups` của đối tượng optimizer trước khi bắt đầu chu trình huấn luyện của từng epoch.

---

## Cell 6 - Chu trình Huấn luyện, Kiểm định và Đánh giá Đơn phát (Test-All)

Trình tự thực thi tiêu chuẩn cho mỗi mini-batch trong chu trình huấn luyện:

```python
optimizer.zero_grad()
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()

```

1. `optimizer.zero_grad()`: Xóa toàn bộ giá trị gradient tích lũy từ bước tính toán trước.
2. `forward pass`: Mô hình xử lý dữ liệu đầu vào `images` và xuất ra giá trị dự báo `logits`.
3. `loss computation`: Hàm `CrossEntropyLoss` tính toán khoảng cách giữa `logits` và nhãn thực tế (`labels`).
4. `backward pass`: Engine Autograd thực hiện lan truyền ngược để tính toán gradient cho từng tham số có `requires_grad=True`.
5. `optimizer.step()`: Thuật toán tối ưu cập nhật giá trị trọng số dựa trên gradient vừa tính toán.

**Quản lý trạng thái mô hình:**
Lệnh `model.train()` kích hoạt trạng thái huấn luyện cho Dropout và Batch Normalization. Lệnh `model.eval()` cùng decorator `@torch.no_grad()` chuyển mô hình sang trạng thái đánh giá, đồng thời ngắt cây tính toán gradient (computation graph) để tối ưu hóa bộ nhớ VRAM.

Trong suốt quá trình huấn luyện, trọng số đạt chỉ số `Validation Accuracy` cao nhất được trích xuất và lưu trữ tạm thời tại `best_state`. Thuật toán Dừng sớm (Early Stopping) theo dõi giá trị `Validation Loss` để ngắt chu trình huấn luyện nếu chỉ số không cải thiện sau một khoảng thời gian chờ (`patience`).

Kết thúc các epoch, hệ thống nạp lại trọng số tối ưu từ `best_state`, thực hiện đánh giá trên tập kiểm thử (`TEST_DATA`) **đúng 01 lần duy nhất**. Kết quả trên tập test hoàn toàn không tham gia vào quá trình lan truyền ngược hay tuyển chọn mô hình.

---

## Cell 7 - Kiểm thử Tính Toàn vẹn Mã nguồn (Sanity Check)

Cell 7 thực hiện khởi tạo một batch dữ liệu giả lập để truy xuất mô hình trên GPU nhằm kiểm tra tính đúng đắn về mặt kỹ thuật trước khi chạy chính thức. Các điều kiện biên cần thỏa mãn:

* Tensor đầu vào đạt kích thước tiêu chuẩn `[batch_size, 3, 32, 32]`.
* Output Logits đạt kích thước `[batch_size, 10]`.
* Giá trị hàm mất mát (Loss) là một số thực hữu hạn (không xuất hiện trạng thái `NaN` hoặc `Inf`).

---

## Cell 8 - Thực thi Thực nghiệm và Trích xuất Nhật ký (Logging Protocol)

Hệ thống sử dụng 02 vòng lặp lồng nhau: Vòng lặp ngoài duyệt qua danh mục cấu hình thực nghiệm, vòng lặp trong duyệt qua các giá trị `seed` (42 và 2026).

Ngay sau khi kết thúc mỗi lượt chạy (`run`), dữ liệu nhật ký sẽ được ghi trực tiếp xuống các tập tin CSV tương ứng trên đĩa cứng. Cơ chế này đảm bảo dữ liệu thô được bảo toàn nguyên vẹn trong trường hợp phiên làm việc Kaggle bị ngắt đột ngột. Tệp tin đầu ra được định danh theo công thức `MEMBER` và `PART` để tránh xung đột dữ liệu.

---

## Cell 9 - Tổng hợp Dữ liệu và Xử lý Thống kê (Data Aggregation)

Hàm `read_matching_csv` truy xuất toàn bộ tập tin nhật ký thô trong thư mục làm việc và các dữ liệu liên kết. Hàm `drop_duplicates(["experiment_id", "seed"])` loại bỏ các bản ghi trùng lặp, chỉ giữ lại lượt chạy chính thức gần nhất.

Hệ thống gom nhóm dữ liệu theo cấu hình thực nghiệm `groupby("experiment_id")` để tính toán các thông số thống kê trên 02 lượt chạy seed:

Test Accuracy Mean = (1 / N) * SUM(Acc_i)

Test Accuracy Std = SQRT( (1 / (N - 1)) * SUM((Acc_i - Mean)^2) )

Cần lưu ý: Giá trị Độ lệch chuẩn Std đại diện cho biến động thực nghiệm giữa các điểm khởi tạo ngẫu nhiên khác nhau, không đóng vai trò là khoảng tin cậy (Confidence Interval) hay kiểm định giả thuyết thống kê.

---

## Cell 10 - Kiểm tra Tính Toàn vẹn của Quy trình Đánh giá (Test-All Verification)

Cell này không thực hiện huấn luyện mà đóng vai trò là bộ kiểm duyệt dữ liệu (validator). Hệ thống sẽ rà soát sự tồn tại của đủ 26 cấu hình thực nghiệm, đảm bảo mỗi cấu hình có đủ 02 lượt chạy seed và không bị khuyết thiếu chỉ số `Test Accuracy`.

**Nguyên tắc toàn vẹn khoa học:**
Quy trình Test-All chỉ đảm bảo tính khách quan khi toàn bộ cấu hình thực nghiệm và siêu tham số đã được đóng băng cố định từ trước. Việc sử dụng kết quả trên tập test để quay lại hiệu chỉnh siêu tham số hoặc lọc bỏ các cấu hình có kết quả kém sẽ làm mất đi tính độc lập của tập kiểm thử, biến tập test thành tập validation và vi phạm nguyên tắc nghiên cứu thực nghiệm.

---

## Danh mục Câu hỏi Trọng tâm Phục vụ Đánh giá Đề tài

1. **Khởi tạo kiến trúc:** Tại sao ResNet cho CIFAR-10 không sử dụng kernel 7 x 7 và lớp Max-pooling ở tầng đầu tiên như bản nguyên bản cho ImageNet?
2. **Cơ chế Residual:** Khối đường tắt (shortcut) giải quyết vấn đề gì trong mạng Nơ-ron sâu? Khi nào bắt buộc phải sử dụng phép tích tụ 1 x 1 tại đường tắt?
3. **Quản lý Gradient:** Tại sao phải gọi lệnh `optimizer.zero_grad()` trước khi thực hiện `loss.backward()`?
4. **Trạng thái Mô hình:** Sự thay đổi hành vi chi tiết của lớp Batch Normalization và Dropout khi chuyển đổi giữa `model.train()` và `model.eval()`?
5. **Đánh giá Dữ liệu:** Tại sao không áp dụng các kỹ thuật tăng cường dữ liệu ngẫu nhiên (Random Augmentation) trên tập Validation và Test?
6. **Độc lập Thực nghiệm:** Mục đích của việc tách biệt `split_seed` (4653) và `training_seed` (42, 2026)?
7. **Nguyên tắc Phân lập Biến số:** Tại sao trong một bài thực nghiệm loại trừ (Ablation Study) chỉ được phép thay đổi duy nhất một yếu tố kỹ thuật tại một thời điểm?
8. **Động lực học LR:** So sánh công thức biến thiên tốc độ học theo thời gian của các chiến lược Warm-up, Step Decay và Cosine Annealing?
9. **Thuật toán Tối ưu:** Sự khác biệt về mặt toán học trong cách xử lý Weight Decay giữa thuật toán Adam và AdamW?
10. **Thống kê Thực nghiệm:** Độ giá trị và giới hạn kết luận của chỉ số Trung bình ± Độ lệch chuẩn Mean ± Std khi thực hiện trên 02 giá trị seed?
11. **Ràng buộc Đánh giá:** Tại sao việc căn cứ vào kết quả trên tập Test để lựa chọn cấu hình mô hình lại vi phạm nghiêm trọng quy trình thực nghiệm Machine Learning?
12. **Truy xuất Mã nguồn:** Trích xuất đoạn mã cụ thể trong notebook chịu trách nhiệm khởi tạo một dòng bản ghi trong tệp `summary_*.csv`?
