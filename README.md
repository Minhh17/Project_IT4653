# IT4653 - Đề tài 3

Nhóm khảo sát các kỹ thuật tối ưu hóa khi huấn luyện một ResNet-18 nhỏ trên dataset CIFAR-10. Toàn bộ phần chạy được đặt trong **một notebook Kaggle**.

## Yêu cầu mã nguồn trong đề bài

Đề bài yêu cầu: GitHub **hoặc** file `.zip`, kèm README, `requirements.txt`, và một notebook/script tái lập kết quả chính. Với project này, từng ý được đáp ứng như sau:

| Yêu cầu | Nhóm | Ý nghĩa |
|---|---|---|
| Repository GitHub hoặc `.zip` | Link GitHub; tạo thêm `.zip` dự phòng trước hạn nộp | Hai cách lưu một thư mục mã nguồn. |
| README cài đặt và chạy lại | File README này | Cài đặt trên Kaggle là import notebook, bật GPU và Add Input CIFAR-10. Không bắt buộc cài local. |
| `requirements.txt` | Danh sách thư viện notebook dùng | Kaggle đã cài sẵn; phiên bản được in ở cell đầu. |
| Notebook/script tái lập kết quả chính | `notebooks/DeTai3_Kaggle.ipynb` | Notebook tự đọc dữ liệu, tạo split, dựng model, train hai seed, ghi CSV, ghép mean ± std và vẽ hình. |
| Nhật ký thí nghiệm | Các CSV trong `results/raw/` | Mỗi lượt chạy có thời gian, cấu hình, seed và kết quả. |

Hướng dẫn tái lập từng bước nằm ở [docs/REPRODUCE_RESULTS.md](docs/REPRODUCE_RESULTS.md).

## Cấu trúc repository

```text
README.md
TEAM_PLAN.md
requirements.txt
notebooks/DeTai3_Kaggle.ipynb
docs/REPRODUCE_RESULTS.md
docs/CODE_WALKTHROUGH_SIMPLE.md
docs/AI_USAGE_DECLARATION.md
results/README.md
```
Khi nộp, bổ sung kết quả thật, báo cáo và slide:

```text
results/
├── raw/                 # summary/epoch/step CSV của ba thành viên
├── mean_std.csv
├── final_test.csv
└── figures/             # ít nhất 6 hình dùng trong báo cáo
report.pdf               # báo cáo 8–12 trang
slides.pdf               # slide bảo vệ 8–12 slide
```

## Quy trình

### 1. Tạo GitHub nhóm

1. Tạo repository GitHub.
2. Đưa các file trong cấu trúc phía trên lên nhánh `main`.
3. Thành viên khác sửa qua branch/commit hoặc gửi notebook cho người tích hợp. Không để ba training loop khác nhau.
4. Khi đã chốt một bản chạy thật, cả nhóm dùng đúng bản đó.

GitHub chỉ dùng để giữ bản chuẩn, lịch sử thay đổi, tài liệu và kết quả nhỏ. Nhóm phát triển và huấn luyện trực tiếp trên Kaggle.

### 2. Đưa notebook lên Kaggle

1. Kaggle → **Create → New Notebook**.
2. Chọn **File → Import Notebook** hoặc upload `notebooks/DeTai3_Kaggle.ipynb`.
3. Mở **Settings → Accelerator → GPU**.
4. Chọn **Add Input** và thêm [CIFAR-10 Python, Version 1](https://www.kaggle.com/datasets/harshajakkam/cifar-10-python-cifar-10-python-tar-gz).
5. Bản dữ liệu phải chứa đúng thư mục:

```text
cifar-10-batches-py/
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
├── test_batch
└── batches.meta
```
### 3. Tự kiểm tra trên Kaggle

Ở cell đầu đặt:

```python
MEMBER = 1
DEBUG = True
PART = "pilot1"
RUN_IDS = []
```

Sau đó chọn **Run All**. Notebook sẽ:

- in Python, PyTorch, torchvision, tên GPU và đường dẫn CIFAR-10;
- tạo split cố định 45.000 train / 5.000 validation;
- in shape của batch và logits;
- chạy forward + backward một batch;
- train một cấu hình, seed 42, trong một epoch trên tập nhỏ;
- lưu ba CSV vào `/kaggle/working/it4653`.

Đây là bước nhóm tự kiểm tra ngay trong môi trường thật. Hãy nhìn loss, accuracy, shape và các file đầu ra; repository không cần một bộ test riêng.

### 4. Chạy thí nghiệm thật

Sau khi pilot hợp lý:

```python
DEBUG = False       # 20 epochs, seeds 42 và 2026
PART = "part1"
RUN_IDS = ["opt_sgd", "opt_nesterov"]
```

Mỗi thành viên dùng **Copy & Edit** từ cùng notebook version:

- `MEMBER=1`: optimizer, normalization và anchor;
- `MEMBER=2`: learning-rate schedule;
- `MEMBER=3`: regularization;
- `MEMBER=0`: toàn bộ 26 cấu hình, chỉ dùng khi muốn tái lập tất cả trên một tài khoản.

`RUN_IDS=[]` chạy toàn bộ phần đã chọn. Nên chia thành `part1`, `part2`... để vừa thời lượng một phiên Kaggle. Mỗi cấu hình chính thức tự chạy hai seed 42 và 2026.

Sau mỗi part, chọn **Save Version** và tải hoặc tạo Kaggle Dataset từ ba file:

```text
summary_memberN_partX.csv
epoch_log_memberN_partX.csv
step_log_memberN_partX.csv
```

Không dùng CSV có `PART="pilot..."` trong báo cáo.
Quy ước file kết quả chi tiết nằm ở [results/README.md](results/README.md).

## Phạm vi thực nghiệm

- 6 optimizer: SGD, SGD momentum, Nesterov, RMSProp, Adam, AdamW.
- Constant/step/cosine × có/không warm-up.
- 8 cấu hình regularization: none, weight decay, ba Dropout, augmentation, early stopping, combined.
- BN/LN/GN × batch size 8/32/128.
- Một anchor dùng chung: 26 cấu hình duy nhất × 2 seed = **52 lượt chạy**.
- Ít nhất 6 biểu đồ và một mục “khuyến nghị thực hành” dựa trên số liệu thật của nhóm.

`≥20 lượt` trong PDF là mức sàn, không thay thế danh sách phép so sánh bắt buộc. PDF yêu cầu đo regularization riêng và kết hợp nhưng không ấn định chính xác tổng số cấu hình. Tài liệu Kickoff lại không nhất quán về cấu hình `WD + augmentation`; nếu giảng viên xác nhận cần cấu hình trung gian này, thêm một dòng và tổng thành 54 lượt.

Label smoothing/Mixup/CutMix và learning-rate range test nằm ở phần mở rộng, nên nhóm có thể bỏ để giữ phạm vi đơn giản.

### Baseline đã triển khai trong notebook

| Thành phần | Giá trị |
|---|---|
| Model | ResNet-18 CIFAR, stem 3×3 stride 1, không max-pool |
| Data | CIFAR-10, split 45k/5k bằng seed 4653 |
| Epoch / training seed | 20 / `{42, 2026}` |
| Batch size | 128, trừ nhánh normalization |
| Optimizer | SGD momentum 0.9 |
| LR / schedule | 0.1 / constant, không warm-up |
| Normalization | BatchNorm |
| Weight decay | `5e-4`, chỉ Conv/Linear weights; không decay bias/norm |
| Dropout / augmentation / early stop | `0.0` / tắt / tắt |

Các LR `0.1` cho họ SGD và `0.001` cho RMSProp/Adam/AdamW trong notebook là **giá trị gợi ý để pilot**, chưa phải kết luận. Ngày 1, thử cùng số ứng viên và cùng ngân sách cho từng optimizer, ghi lại lựa chọn, rồi mới khóa LR và `NOTEBOOK_VERSION` cho official runs.


## Nguồn tham khảo và phần nhóm tự viết

Notebook dùng API huấn luyện của PyTorch nhưng tự định nghĩa ResNet-18 cho ảnh 32×32, normalization, training loop, danh sách ablation, log và biểu đồ. Nguồn lý thuyết tối thiểu nên trích dẫn trong README/report gồm:

- Kaiming He và cộng sự, [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
- Alex Krizhevsky, [CIFAR-10 and CIFAR-100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html).
- Tài liệu chính thức của [PyTorch optimizers](https://pytorch.org/docs/stable/optim.html) và [normalization layers](https://pytorch.org/docs/stable/nn.html#normalization-layers).

Nếu nhóm lấy thêm code/công thức từ nguồn công khai, phải thêm link và nói rõ cell/phần nào được tham khảo. Việc AI hỗ trợ bản nháp được khai báo theo [docs/AI_USAGE_DECLARATION.md](docs/AI_USAGE_DECLARATION.md); nhóm vẫn phải đọc, chạy và giải thích được mọi dòng.

Chi tiết phân công nằm trong [TEAM_PLAN.md](TEAM_PLAN.md).