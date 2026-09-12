# IT4653 - Đề tài 3: Tối ưu hóa quá trình huấn luyện ResNet-18 trên tập dữ liệu CIFAR-10

## 1. Tổng quan Dự án

Dự án thực hiện khảo sát và đánh giá thực nghiệm các kỹ thuật tối ưu hóa trong quá trình huấn luyện mô hình ResNet-18 (phiên bản điều chỉnh kích thước cho ảnh 32x32) trên tập dữ liệu CIFAR-10. 

Môi trường thực thi được chuẩn hóa toàn bộ trên **Kaggle Notebook (sử dụng GPU Accelerator)**. Thiết kế này đảm bảo tính đóng gói, cho phép các thành viên trong nhóm phát triển, thực thi và kiểm chứng kết quả trực tiếp theo thời gian thực mà không phụ thuộc vào hạ tầng phần cứng cục bộ.

Để tối ưu hóa cấu trúc mã nguồn và tập trung vào mục tiêu nghiên cứu, dự án không triển khai các đóng gói phức tạp (Python packages, YAML configs, hay test-runner scripts). Toàn bộ luồng xử lý (pipeline) từ nạp dữ liệu, định nghĩa mô hình, huấn luyện, ghi log đến trực quan hóa đều được tích hợp trong một Notebook duy nhất.

---

## 2. Quy chuẩn Đáp ứng Yêu cầu Mã nguồn & Tái lập (Reproducibility)

Tất cả các thành phần nộp bài được đối chiếu trực tiếp với yêu cầu của đề tài:

| Yêu cầu đề bài | Thành phần bàn giao | Mô tả kỹ thuật |
| :--- | :--- | :--- |
| **Mã nguồn** | Repository GitHub & File lưu trữ `.zip` | Mã nguồn đồng bộ trên GitHub; bản đóng gói `.zip` được chuẩn bị sẵn làm phương án dự phòng. |
| **Hướng dẫn cài đặt** | File `README.md` | Môi trường mặc định là Kaggle GPU (đã tích hợp sẵn CUDA, PyTorch và tập dữ liệu CIFAR-10). |
| **Phụ thuộc thư viện** | File `requirements.txt` | Trích xuất chính xác phiên bản các thư viện được khởi tạo tại phiên chạy chuẩn trên Kaggle. |
| **Kịch bản tái lập** | Notebook `notebooks/DeTai3_Kaggle.ipynb` | Khởi tạo pipeline khép kín: xử lý dữ liệu, phân chia tập dữ liệu, định nghĩa kiến trúc, huấn luyện đa-seed, ghi log CSV và xuất biểu đồ. |
| **Nhật ký thực nghiệm** | Thư mục `results/raw/` | Lưu trữ nhật ký chạy chi tiết (thời gian, thông số cấu hình, seed, chỉ số từng epoch). Mọi số liệu trong báo cáo được trích xuất trực tiếp từ đây. |

### Định nghĩa tính Tái lập (Reproducibility Protocol)

Tiêu chuẩn tái lập của dự án không yêu cầu thực thi lại toàn bộ 52 lượt chạy (runs) trong một phiên làm việc đơn lẻ. Thay vào đó, hệ thống cung cấp đầy đủ mã nguồn, tham số cấu hình, hạt giống ngẫu nhiên (seeds) và quy trình chuẩn để:

1. **Kiểm thử luồng (Pilot Run):** Xác nhận tính đúng đắn của pipeline trong thời gian ngắn.
2. **Tái lập độc lập:** Chạy lại bất kỳ cấu hình cụ thể nào trên 2 seeds ngẫu nhiên.
3. **Tổng hợp dữ liệu:** Tự động kết xuất bảng giá trị trung bình kèm độ lệch chuẩn (`Mean ± Std`) và hệ thống 06 biểu đồ cốt lõi từ tập dữ liệu thô (`.csv`).

*Chi tiết các bước thực thi được trình bày tại [docs/REPRODUCE_RESULTS.md](docs/REPRODUCE_RESULTS.md) và diễn giải chi tiết mã nguồn tại [docs/CODE_WALKTHROUGH_SIMPLE.md](docs/CODE_WALKTHROUGH_SIMPLE.md).*

---

## 3. Cấu trúc Repository

```text
.
├── README.md                           # Tóm tắt dự án & hướng dẫn tổng quan
├── TEAM_PLAN.md                        # Phân công nhiệm vụ & đóng góp của thành viên
├── requirements.txt                    # Thông số phiên bản môi trường thực thi
├── notebooks/
│   └── DeTai3_Kaggle.ipynb             # Notebook huấn luyện & tổng hợp chính
├── docs/
│   ├── REPRODUCE_RESULTS.md            # Quy trình chi tiết tái lập kết quả
│   ├── CODE_WALKTHROUGH_SIMPLE.md      # Giải thích cấu trúc mã nguồn
│   └── AI_USAGE_DECLARATION.md         # Khai báo mức độ sử dụng công cụ AI
└── results/                            # Kết quả thực nghiệm
    ├── README.md                       # Quy chuẩn định dạng file log
    ├── mean_std.csv                    # Dữ liệu tổng hợp (Mean ± Std)
    ├── raw/                            # Log dữ liệu thô theo từng phiên chạy
    └── figures/                        # Trực quan hóa kết quả (tối thiểu 6 đồ thị)

```

*Lưu ý:* Các dữ liệu tạm (checkpoints mô hình, dữ liệu thô CIFAR-10, Kaggle API Tokens) bị loại bỏ khỏi hệ thống quản lý phiên bản Git. Chỉ số trên tập Kiểm thử (Test set) được tính toán ngay sau mỗi lượt huấn luyện, do đó việc lưu trữ 52 file trọng lượng mô hình (checkpoints) là không cần thiết.

---

## 4. Quy trình Thực thi & Khai thác Thí nghiệm

### 4.1 Khởi tạo Môi trường (Kaggle Environment Setup)

1. Khởi tạo một Notebook mới trên Kaggle và nhập (import) file `notebooks/DeTai3_Kaggle.ipynb`.
2. Bật môi trường tính toán **GPU (NVIDIA T4 hoặc tương đương)**.
3. Liên kết tập dữ liệu [CIFAR-10 Python Dataset](https://www.kaggle.com/datasets/pankrzysiu/cifar10-python) vào môi trường làm việc. Hệ thống tự động nhận diện cấu trúc tệp `.tar.gz` hoặc thư mục đã giải nén `cifar-10-batches-py/`.

### 4.2 Lượt chạy Kiểm thử Pipeline (Pilot Run)

Để kiểm tra tính toàn vẹn của mã nguồn trước khi huấn luyện chính thức, thiết lập biến cấu hình tại cell khởi tạo:

```python
RUN_ALL_CONFIGS = False
MEMBER = 1
DEBUG = True
PART = "pilot1"
RUN_IDS = []

```

Thực thi `Run All`. Hệ thống sẽ tự động:

* Kiểm tra thông số phần cứng, các thư viện phụ thuộc và đường dẫn dữ liệu.
* Thực hiện phân chia tập dữ liệu chuẩn: 45,000 ảnh Huấn luyện (Train), 5,000 ảnh Xác thực (Validation), và 10,000 ảnh Kiểm thử (Test).
* Kiểm tra kích thước tensor (shape) và tính toán luồng Forward/Backward trên một batch.
* Thực thi 01 epoch với 01 cấu hình mẫu trên tập dữ liệu rút gọn để xác nhận luồng ghi dữ liệu output (`.csv`).

### 4.3 Thực thi Thí nghiệm Chính thức (Official Experiments)

Sau khi hoàn tất bước kiểm thử, chuyển trạng thái `DEBUG = False` để huấn luyện trên 20 epochs với 02 hạt giống ngẫu nhiên cố định (`seeds = [42, 2026]`):

```python
DEBUG = False 
PART = "part1"
RUN_IDS = ["opt_sgd", "opt_nesterov"]

```

Nhóm phân chia phạm vi nghiên cứu theo cấu trúc:

* `MEMBER = 1`: Nhóm thuật toán Tối ưu hóa (Optimizers), Chuẩn hóa (Normalization) và Cấu hình Anchor.
* `MEMBER = 2`: Nhóm Lịch trình Thay đổi Tốc độ học (Learning Rate Schedules).
* `MEMBER = 3`: Nhóm Kỹ thuật Điều hòa (Regularization).
* `MEMBER = 0` / `RUN_ALL_CONFIGS = True`: Thực thi toàn bộ 26 cấu hình thực nghiệm.

**Cơ chế đánh giá:** Cuối mỗi lượt huấn luyện, điểm dừng tối ưu được lựa chọn dựa trên chỉ số Accuracy cao nhất trên tập Validation. Trọng số tại điểm này được dùng để đánh giá duy nhất **01 lần** trên tập Test và kết quả được ghi trực tiếp vào log `summary.csv`.

---

## 5. Phương pháp Phân tích & Tích hợp Dữ liệu

Sau khi các thành viên hoàn tất các phần thí nghiệm riêng lẻ:

1. Tải các file nhật ký thô (`summary_memberN_partX.csv`, `epoch_log_...`, `step_log_...`) lên môi trường Kaggle chung.
2. Thiết lập chế độ xử lý dữ liệu tổng hợp trong Notebook: `DEBUG = False` và gọi cell **"Ghép CSV và vẽ tối thiểu 6 biểu đồ"** (bỏ qua các cell huấn luyện).
3. Hệ thống sẽ tự động tính toán giá trị trung bình, độ lệch chuẩn và xuất tập tin `mean_std.csv` cùng 06 biểu đồ phân tích hiệu năng vào thư mục `results/figures/`.

---

## 6. Phạm vi Thực nghiệm (Experimental Matrix)

Thực nghiệm được thiết kế nhằm đánh giá toàn diện ảnh hưởng của các thành phần lên quá trình huấn luyện:

* **Optimizers (06):** SGD, SGD Momentum, Nesterov, RMSprop, Adam, AdamW.
* **LR Schedules:** Constant, Step Decay, Cosine Annealing (kết hợp trạng thái Có/Không Warm-up).
* **Regularization (08):** Baseline, Weight Decay, Dropout (3 mức độ), Data Augmentation, Early Stopping, và Combined Regularization.
* **Normalization & Batch Size:** BatchNorm, LayerNorm, GroupNorm kết hợp với Batch Sizes {8, 32, 128}.
* **Quy mô Thực nghiệm:** 26 cấu hình độc lập x 2 seeds = **52 lượt huấn luyện (runs)**.

### Thông số Mô hình Cơ sở (Baseline Configuration)

| Thành phần | Thông số kỹ thuật |
| --- | --- |
| **Kiến trúc** | ResNet-18 (Chỉnh sửa stem 3x3, stride 1, bỏ MaxPool ban đầu) |
| **Tập dữ liệu** | CIFAR-10 (45k Train / 5k Val, cố định Split Seed = 4653) |
| **Huấn luyện** | 20 Epochs | Batch Size = 128 | Seeds = {42, 2026} |
| **Thuật toán Tối ưu** | SGD với Momentum = 0.9, Learning Rate gốc = 0.1 |
| **Tối ưu hóa khác** | BatchNorm | Weight Decay = 5e-4 (chỉ áp dụng cho Weights) |

---

## 7. Nguyên tắc Nguyên vẹn Thực nghiệm (Experimental Integrity)

Để đảm bảo tính khoa học và minh bạch của kết quả nghiên cứu, dự án tuân thủ nghiêm ngặt các quy tắc:

1. **Cố định hằng số:** Toàn bộ các lượt chạy sử dụng chung tập phân chia dữ liệu (Seed 4653) và các hạt giống ngẫu nhiên huấn luyện (Seeds 42, 2026). Tập Validation không áp dụng các kỹ thuật Tăng cường dữ liệu (Data Augmentation).
2. **Nguyên tắc Đơn biến (Controlled Experiments):** Trong mỗi phép so sánh, chỉ thay đổi duy nhất nhân tố cần khảo sát và giữ nguyên ngân sách Epoch.
3. **Thống kê Chuẩn xác:** Mọi chỉ số báo cáo đều diễn giải dưới dạng `Mean ± Sample Standard Deviation`. Trường hợp 2 seeds cho xu hướng trái ngược, kết quả được phân loại là "Chưa đủ bằng chứng thống kê".
4. **Khóa Cấu hình (Protocol Freezing):** Khóa toàn bộ tham số, siêu tham số và phiên bản mã nguồn trước khi đánh giá trên tập Test. Tuyệt đối không thay đổi thông số hệ thống dựa trên kết quả của tập Test.
5. **Truy xuất Nguồn gốc:** Đánh giá đúng 01 lần trên tập Test tại mốc thu được điểm Validation tốt nhất. Không thực hiện đánh giá lại trên Epoch cuối để chọn lọc số liệu.
6. **Tính Trung thực của Dữ liệu:** Không can thiệp thủ công hoặc chỉnh sửa số liệu nhật ký. Bảng biểu và đồ thị phân tích được sinh hoàn toàn tự động bằng mã nguồn từ tập dữ liệu thô.

---

## 8. Tài liệu Tham khảo

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
2. PyTorch Documentation: [Optimization Modules](https://pytorch.org/docs/stable/optim.html) & [Normalization Layers](https://pytorch.org/docs/stable/nn.html#normalization-layers).
3. Khai báo chi tiết về việc ứng dụng công cụ AI hỗ trợ trong quá trình biên soạn mã nguồn và tài liệu được lưu trữ tại [docs/AI_USAGE_DECLARATION.md](https://www.google.com/search?q=docs/AI_USAGE_DECLARATION.md).

---

## 9. Danh mục Kiểm tra Hoàn tất (Submission Checklist)

* [x] Repository GitHub thiết lập quyền truy cập công khai (Public) kèm bản nộp dự phòng `.zip`.
* [x] Đã thực hiện kiểm thử độc lập quy trình cài đặt và tái lập trên một tài khoản Kaggle sạch.
* [x] Tệp `requirements.txt` phản ánh chính xác môi trường thực thi chuẩn.
* [x] Mã nguồn công khai đặt trạng thái `DEBUG = True` mặc định, không chứa các thông tin xác thực/API Keys.
* [x] Thư mục `results/raw/` lưu trữ đầy đủ log dữ liệu của 52 lượt chạy chính thức.
* [x] Tệp `TEAM_PLAN.md` làm rõ vai trò và tỷ lệ đóng góp của từng thành viên (Tổng = 100%).
