# Quy trình Tái lập Kết quả Thực nghiệm trên Kaggle

Tài liệu này quy định quy trình chuẩn (standard protocol) nhằm đảm bảo khả năng tái lập kết quả thực nghiệm (reproducibility) cho các thành viên trong nhóm nghiên cứu và hội đồng đánh giá. Môi trường thực thi được hỗ trợ chính thức là **Kaggle Notebook (cấu hình GPU)**; không yêu cầu thiết lập môi trường cục bộ (local environment).

---

## 1. Phân cấp thực nghiệm (Experimental Levels)

Quy trình tái lập được chia thành 03 mức độ kiểm thử:

### Mức A: Kiểm thử toàn bộ pipeline (Sanity Check)
* **Mục đích:** Kiểm tra tính toàn vẹn của luồng xử lý dữ liệu (data pipeline), trạng thái GPU, khởi tạo mô hình, quá trình lan truyền tiến/lùi (forward/backward pass), vòng lặp huấn luyện/đánh giá (train/val loop) và xuất dữ liệu CSV.

```python
MEMBER = 1
DEBUG = True
PART = "pilot_reproduce"
RUN_IDS = []
RUN_ALL_CONFIGS = False

```

* **Yêu cầu thực thi:** Chạy toàn bộ notebook theo thứ tự từ trên xuống. Kết quả ở chế độ `pilot` chỉ phục vụ mục đích kiểm lỗi mã nguồn (debugging), không đọc tập kiểm thử (test set) và không được ghi nhận vào báo cáo chính thức.

---

### Mức B: Tái lập một cấu hình đơn lẻ (Single Experiment Reproduce)

* **Mục đích:** Tái lập một phép so sánh cụ thể được trích dẫn trong báo cáo (ví dụ: so sánh mô hình cơ sở Anchor và tối ưu hóa AdamW).

```python
MEMBER = 1
DEBUG = False
PART = "reproduce_optimizer"
RUN_IDS = ["anchor", "opt_adamw"]

```

* **Yêu cầu thực thi:** Mỗi cấu hình sẽ tự động thực thi với 02 giá trị ngẫu nhiên (seed): `42` và `2026`. Sau khi hoàn tất, cell tổng hợp sẽ tính toán giá trị Trung bình ± Độ lệch chuẩn Mean ± Std từ 04 lượt chạy. Cấu hình `RUN_IDS` có thể thay đổi linh hoạt theo ID của biểu đồ hoặc bảng dữ liệu cần kiểm chứng.

---

### Mức C: Tái lập toàn bộ không gian thực nghiệm (Full Grid Reproduce)

```python
RUN_ALL_CONFIGS = True

```

* **Yêu cầu thực thi:** Khi gán `RUN_ALL_CONFIGS = True`, hệ thống sẽ tự động ghi đè các tham số thiết lập toàn bộ tập thực nghiệm 26 cấu hình x 2 seeds = 52 lượt chạy, kèm theo đo lường tự động trên tập test.
* **Lưu ý về giới hạn hạ tầng:** Trên phần cứng Kaggle Tesla T4, tổng thời gian thực thi dự kiến sẽ vượt quá giới hạn 12 giờ của tính năng *Save & Run All*. Chế độ này chỉ áp dụng khi sử dụng phần cứng cao cấp hơn hoặc môi trường không giới hạn thời gian phiên làm việc (session duration limit). Để phân chia các khối thực nghiệm (`RUN_IDS`) trên môi trường T4, bắt buộc duy trì `RUN_ALL_CONFIGS = False` và phân tải qua cấu hình `MEMBER` / `PART`.

---

## 2. Khởi tạo môi trường thực thi sạch (Clean Environment Setup)

1. Tải tập tin `notebooks/DeTai3_Kaggle.ipynb` từ đúng `commit/release` được chỉ định trên GitHub repository của đề tài.
2. Tại giao diện Kaggle: Chọn **Create** → **New Notebook** → **Import** tập tin notebook đã tải.
3. Thiết lập **Settings** → **Accelerator** → **GPU T4 x2**.
4. Chọn **Add Input** → Tích hợp tập dữ liệu [CIFAR-10 Python](https://www.kaggle.com/datasets/pankrzysiu/cifar10-python) (hoặc đúng phiên bản dữ liệu ghi trong báo cáo).
5. Dữ liệu đầu vào chứa tập tin `cifar-10-python.tar.gz`. Notebook sẽ tự động giải nén và kiểm tra sự tồn tại của cấu trúc `cifar-10-batches-py/data_batch_1` trước khi huấn luyện.
6. Hạn chế sử dụng lệnh `pip install` do các thư viện nền tảng đã được tích hợp sẵn trong Kaggle Docker Image. Trong trường hợp phát sinh lỗi thiếu thư viện, chỉ cài đặt đúng danh mục và phiên bản quy định tại `requirements.txt`.

> **Mô tả phản hồi hệ thống:** Cell khởi tạo sẽ chủ động ngắt thực thi và xuất thông báo lỗi nếu: Không phát hiện GPU, phần cứng được chọn là GPU P100, hoặc cấu trúc dữ liệu đầu vào không khớp với định dạng CIFAR-10 Python tiêu chuẩn.

---

## 3. Danh mục Siêu tham số điều khiển (Execution Parameters)

Trong các lượt chạy thực nghiệm thông thường, người sử dụng chỉ được phép hiệu chỉnh các biến khai báo tại phần khởi tạo notebook:

```python
MEMBER = 1
DEBUG = True
PART = "pilot1"
RUN_IDS = []
RUN_ALL_CONFIGS = False
NOTEBOOK_VERSION = "v2"
SAVE_CHECKPOINTS = False

```

* `MEMBER`: Chỉ định tập thực nghiệm phân công cho thành viên (0 tương ứng với toàn bộ các tập).
* `DEBUG`: `True` áp dụng lấy mẫu nhỏ để kiểm thử nhanh; `False` thực thi chính thức với 20 epochs trên 2 seeds.
* `PART`: Định danh phân đoạn thực nghiệm nhằm tránh ghi đè dữ liệu đầu ra giữa các phiên.
* `RUN_IDS`: Danh sách mã cấu hình thực thi (để rỗng tương ứng với chạy toàn bộ danh mục trong `PART`).
* `RUN_ALL_CONFIGS`: Flag kích hoạt chế độ tự động thực thi 26 cấu hình x 2 seeds (sẽ ghi đè các biến `MEMBER`, `DEBUG`, `PART`, `RUN_IDS`).
* `NOTEBOOK_VERSION`: Phiên bản mã nguồn lõi đã được đồng bộ trong nhóm.
* `SAVE_CHECKPOINTS`: Khuyến nghị thiết lập `False`. Trọng số tối ưu (best checkpoint) sẽ được đánh giá trực tiếp trên tập test trong cùng lượt chạy và tự động giải phóng bộ nhớ ngay sau đó.

*(Tuyệt đối không thay đổi `BASE_CONFIG`, phương pháp phân chia dữ liệu, kiến trúc mô hình hoặc vòng lặp huấn luyện giữa các lượt chạy seed khác nhau trong cùng một cấu hình).*

---

## 4. Chuẩn hóa Dữ liệu và Điểm khởi tạo Ngẫu nhiên (Data & Seeding Protocol)

* **Tập dữ liệu:** CIFAR-10 chuẩn bao gồm 50,000 ảnh huấn luyện và 10,000 ảnh kiểm thử.
* **Phân chia dữ liệu:** Dùng cố định `seed = 4653` để chia tập 50,000 ảnh thành: **45,000 ảnh Training** và **5,000 ảnh Validation**.
* **Seed ngẫu nhiên:** Hai giá trị seed huấn luyện chính thức được cố định là `42` và `2026`.
* **Tăng cường dữ liệu (Data Augmentation):** Chỉ áp dụng biến đổi dữ liệu trên tập train khi cấu hình thực nghiệm yêu cầu cụ thể.
* **Tập Validation & Test:**
* Luôn áp dụng phép biến đổi chuẩn hóa cố định (clean transform), không thực hiện tăng cường dữ liệu và thiết lập `shuffle = False`.
* Checkpoint được chọn là phiên bản đạt **Độ chính xác cao nhất (Best Accuracy)** trên tập Validation trong suốt chu kỳ huấn luyện.
* Sau khi hoàn tất quá trình huấn luyện, mô hình sẽ nạp lại checkpoint tối ưu để đánh giá trên tập Test **đúng 01 lần duy nhất**. Tập Test hoàn toàn không tham gia vào quá trình lan truyền ngược (backward pass), dừng sớm (early stopping) hay tuyển chọn checkpoint.
* Kích thước lô (Batch size) cho quá trình đánh giá (Evaluation) được cố định là `256`, trong khi batch size huấn luyện có thể thay đổi tùy cấu hình (`8`, `32`, `128`).



> **Nguyên tắc kiểm soát biến số:** Đảm bảo tính minh bạch của phép so sánh bằng cách đồng bộ tập dữ liệu, mô hình nền, điểm khởi tạo ngẫu nhiên và tài nguyên tính toán; chỉ thay đổi duy nhất yếu tố kỹ thuật cần khảo sát.

---

## 5. Cấu trúc Nhật ký Đầu ra (Logging Structure)

Để phòng ngừa mất mát dữ liệu do ngắt kết nối phiên làm việc trên Kaggle, hệ thống sẽ tự động xuất nhật ký sau mỗi lượt chạy vào đường dẫn:

/kaggle/working/it4653/
├── summary_memberN_partX.csv
├── epoch_log_memberN_partX.csv
├── step_log_memberN_partX.csv
├── mean_std.csv
└── figures/


* **`summary_*.csv`:** Lưu trữ kết quả của từng lượt chạy `(experiment_id, seed)`, bao gồm: Siêu tham số thực tế, Best Val Accuracy/Epoch, Test Loss/Accuracy, kết quả epoch cuối, thời gian huấn luyện/đánh giá, thông tin GPU, phiên bản thư viện và `NOTEBOOK_VERSION`. Dữ liệu này là cơ sở phục vụ đối soát.
* **`epoch_log_*.csv`:** Ghi nhận thông số theo từng epoch (Loss/Accuracy của Train & Val, Learning Rate, Thời gian). Phục vụ vẽ đường cong hội tụ (Convergence Curves).
* **`step_log_*.csv`:** Ghi nhận Loss và Learning Rate sau mỗi 20 bước tối ưu (optimizer steps). Phục vụ vẽ biểu đồ biến thiên theo bước huấn luyện.
* **`mean_std.csv` & `figures/`:** Kết xuất tự động từ dữ liệu thô (raw logs), không qua xử lý thủ công. Đây là nguồn dữ liệu trực tiếp để trích xuất bảng biểu và đồ thị trong báo cáo.

---

## 6. Quy trình Thực nghiệm Song song (Parallel Execution Standard)

Tất cả thành viên nghiên cứu phải xuất phát từ cùng một mã nguồn `NOTEBOOK_VERSION`.

| Thành viên | Biến `MEMBER` | Phân công nhóm thực nghiệm |
| --- | --- | --- |
| **Thành viên 1** | `1` | Baseline Anchor, Optimizer, Normalization |
| **Thành viên 2** | `2` | Learning-Rate Scheduling |
| **Thành viên 3** | `3` | Regularization |

**Trình tự thực thi cho từng thành viên:**

1. Thực thi kiểm thử ở chế độ `DEBUG = True`.
2. Chuyển sang chế độ chính thức `DEBUG = False`.
3. Phân chia `RUN_IDS` thành các `PART` nhỏ (nếu cần thiết để tối ưu thời gian phiên).
4. Thực hiện **Save Version** sau khi hoàn thành mỗi `PART`.
5. Tải tập tin CSV hoặc đóng gói thành Kaggle Private Dataset.
6. Chuyển giao các tập tin CSV thô cho thành viên tổng hợp.

> **Quy định về đóng góp mã nguồn:** Không tự ý hiệu chỉnh vòng lặp huấn luyện trên các bản sao cá nhân. Trường hợp phát sinh lỗi trong mã nguồn lõi, phải thực hiện cập nhật tập trung, nâng cấp chỉ số `NOTEBOOK_VERSION` và đánh giá lại tính hiệu lực của các lượt chạy trước đó.

---

## 7. Tổng hợp Dữ liệu và Tái tạo Đồ thị (Data Aggregation & Artifact Generation)

1. Khởi tạo một Kaggle Notebook tổng hợp từ bản mã nguồn chuẩn.
2. Chọn **Add Input** toàn bộ các tập tin CSV chính thức thu thập từ các thành viên (Không tích hợp các log ở chế độ `DEBUG/Pilot`).
3. Thiết lập `DEBUG = False` và `NOTEBOOK_VERSION` tương thích với dữ liệu log chính thức, sau đó chạy cell khởi tạo.
4. Thực thi cell: *"Ghép CSV và vẽ tối thiểu 6 biểu đồ"*.
5. Kiểm tra trường dữ liệu `seeds` trong `mean_std.csv`, đảm bảo đạt đủ 2 lượt chạy cho tất cả cấu hình.
6. Tải về tập tin `mean_std.csv` và thư mục `figures/`.
7. Lưu trữ cấu trúc thư mục dự án:
* Tập tin log thô → `results/raw/`
* Bảng tổng hợp → `results/mean_std.csv`
* Đồ thị → `results/figures/`



**Danh mục 06 đồ thị tiêu chuẩn:**

1. Loss huấn luyện/kiểm định theo Epoch của các thuật toán Optimizer.
2. Độ chính xác trên tập Test Mean ± Std của các thuật toán Optimizer.
3. Loss huấn luyện theo Global Step của các chiến lược Learning-Rate Schedule.
4. Độ chính xác trên tập Test Mean ± Std của Learning-Rate Schedule.
5. Tương quan giữa Normalization x Batch Size đến Độ chính xác tập Test.
6. Độ chính xác trên tập Test Mean ± Std của các phương pháp Regularization.

*(Mọi thay đổi về số liệu trong báo cáo phải được tái tạo trực tiếp qua mã nguồn xử lý log, không hiệu chỉnh thủ công trên tập tin kết quả).*

---

## 8. Chuẩn đánh giá Test-All (Test-All Evaluation Protocol)

Trước khi thực thi chính thức, toàn bộ 26 cấu hình, 02 giá trị seed, tốc độ học (LR) và mã nguồn `v2` phải được đóng băng (freeze) trên GitHub Repository. Đối với mỗi cặp `(experiment_id, seed)`, quy trình đánh giá tuân thủ chuỗi xử lý nghiêm ngặt:

Train 45k → Val 5k [Chọn Best Epoch] → Load Best Checkpoint → Test 10k [Đánh giá 1 lần]

* **Tối ưu hóa checkpoint:** Tiêu chí chọn trọng số dựa trên Validation Accuracy lớn nhất. Trường hợp có nhiều epoch đạt giá trị bằng nhau, ưu tiên epoch xuất hiện trước.
* **Đánh giá tập Test:** Bắt buộc áp dụng `model.eval()` và `torch.no_grad()`, không tăng cường dữ liệu, không xáo trộn dữ liệu (`shuffle = False`), kích thước lô đánh giá cố định 256. Nghiêm cấm hành vi đánh giá thêm trọng số ở epoch cuối để chọn lọc kết quả tối ưu hơn.
* **Mục đích của Test-All:** Nhằm đánh giá khả năng tổng quát hóa (generalization) **trong nội bộ từng nhóm khảo sát** so với mô hình cơ sở Anchor. Không sử dụng kết quả này để xếp hạng toàn bộ 26 cấu hình như một cuộc thi công khai, do mỗi nhóm thực nghiệm hướng đến giải quyết các câu hỏi khoa học khác nhau. Nếu xuất hiện sự lệch hướng giữa tập Validation và tập Test, phải ghi nhận cả hai chỉ số và đưa vào phần thảo luận, không thực hiện thay đổi siêu tham số để chạy lại có chọn lọc.
* **Xử lý dữ liệu cũ:** Các dữ liệu log thuộc phiên bản `v1` không chứa thông tin đánh giá tập Test và trọng số tối ưu. Do đó, không gộp dữ liệu `v1` vào bảng báo cáo chính thức `v2`.

---

## 9. Đóng băng Phiên bản Môi trường (Environment Freezing)

Khi thực thi phiên chính thức, cell đầu tiên sẽ xuất thông số hệ thống của Python, PyTorch, Torchvision và GPU. Trích xuất chính xác các phiên bản này để cập nhật vào `requirements.txt`:

torch==<phien_ban_truc_xuat>
torchvision==<phien_ban_truc_xuat>
numpy==<phien_ban_truc_xuat>
pandas==<phien_ban_truc_xuat>
matplotlib==<phien_ban_truc_xuat>


*(Thay thế `<phien_ban_truc_xuat>` bằng thông số thực tế từ phiên làm việc trên Kaggle. Đồng thời ghi nhận `NOTEBOOK_VERSION`, Git Commit SHA và phiên bản Kaggle Dataset trong báo cáo hoặc tài liệu README).*

---

## 10. Quy trình Kiểm thử Tái lập độc lập (Independent Peer Audit)

Một thành viên không tham gia trực tiếp vào quá trình viết mã nguồn notebook sẽ thực hiện quy trình kiểm thử độc lập trước thời hạn nộp báo cáo:

1. Truy cập liên kết GitHub Repository từ một thiết bị hoặc phiên làm việc độc lập.
2. Tải bản sao mã nguồn notebook mới nhất.
3. Import tập tin vào môi trường Kaggle, kích hoạt cấu hình GPU và liên kết dữ liệu đầu vào (Add Input).
4. Thực thi toàn bộ **Mức A (Sanity Check)** từ đầu đến cuối.
5. Xác nhận tính đầy đủ của 03 tập tin nhật ký CSV đầu ra.
6. Trích xuất ngẫu nhiên một tập tin CSV để đối soát các thông số `seed`, `config` và `metrics`.
7. Liên kết dữ liệu log thô chính thức và thực thi quy trình tái tạo bảng `mean_std.csv` cùng 06 đồ thị tiêu chuẩn.

Nếu quy trình trên được thực hiện thành công và không phát sinh bất kỳ sự cố kỹ thuật nào, mã nguồn và tài liệu hướng dẫn được xác nhận **Đạt tiêu chuẩn tái lập nghiên cứu (Reproducible Standard)**.
