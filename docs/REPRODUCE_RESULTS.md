# Hướng dẫn tái lập kết quả trên Kaggle

Tài liệu này là “đường chạy chuẩn” dành cho thành viên trong nhóm và người chấm. Môi trường được nhóm hỗ trợ chính thức là **Kaggle Notebook có GPU**; không cần chạy local.

## 1. Tái lập gồm ba mức

### Mức A - kiểm tra toàn bộ pipeline trong vài phút

Mục đích: kiểm tra dataset, GPU, model, forward/backward, train, validation và ghi CSV đều hoạt động.

```python
MEMBER = 1
DEBUG = True
PART = "pilot_reproduce"
RUN_IDS = []
```

Chạy notebook từ trên xuống. Kết quả pilot chỉ để kiểm tra code, không đưa vào bảng báo cáo.

### Mức B - chạy lại một kết quả chính

Mục đích: tái lập một phép so sánh được dùng trong báo cáo. Ví dụ chạy lại anchor và AdamW:

```python
MEMBER = 1
DEBUG = False
PART = "reproduce_optimizer"
RUN_IDS = ["anchor", "opt_adamw"]
```

Mỗi cấu hình tự chạy seed 42 và 2026. Sau khi xong, cell tổng hợp tạo mean ± std từ bốn lượt chạy. Có thể thay `RUN_IDS` bằng ID của hình/bảng cần kiểm chứng.

### Mức C - chạy lại toàn bộ ma trận

```python
MEMBER = 0
DEBUG = False
PART = "full"
RUN_IDS = []
```

Chế độ này gồm 26 cấu hình × 2 seed = 52 lượt. Nó có thể vượt thời lượng một phiên Kaggle, nên thực tế nhóm chia cho ba tài khoản hoặc chia `RUN_IDS` thành nhiều part. Dù chia phiên, tất cả vẫn dùng cùng notebook, seed và baseline.

## 2. Chuẩn bị một Kaggle Notebook sạch

1. Tải `notebooks/DeTai3_Kaggle.ipynb` từ đúng release/commit GitHub mà nhóm ghi trong báo cáo.
2. Kaggle → **Create → New Notebook** → import file notebook.
3. Settings → Accelerator → chọn GPU.
4. Add Input → thêm [CIFAR-10 Python, Version 1](https://www.kaggle.com/datasets/harshajakkam/cifar-10-python-cifar-10-python-tar-gz), hoặc đúng dataset/version mà nhóm đã ghi trong báo cáo.
5. Dataset phải chứa `cifar-10-batches-py/data_batch_1` tới `data_batch_5`, `test_batch` và `batches.meta`.
6. Không cần chạy `pip install` vì các thư viện đã có trong Kaggle image. Nếu import báo thiếu thư viện, chỉ cài đúng package/phiên bản ghi trong `requirements.txt`.

Cell đầu sẽ dừng với thông báo rõ nếu không thấy GPU hoặc không tìm thấy đúng một thư mục `cifar-10-batches-py`.

## 3. Các giá trị được phép sửa

Trong lần chạy thông thường, chỉ sửa sáu giá trị ở đầu notebook:

```python
MEMBER = 1
DEBUG = True
PART = "pilot1"
RUN_IDS = []
NOTEBOOK_VERSION = "v1"
SAVE_CHECKPOINTS = False
```

- `MEMBER`: chọn danh sách thí nghiệm của thành viên; `0` là tất cả.
- `DEBUG`: `True` chạy nhanh trên tập nhỏ; `False` chạy chính thức 20 epoch và hai seed.
- `PART`: tên phần để các phiên không ghi đè nhau.
- `RUN_IDS`: danh sách cấu hình cần chạy; rỗng nghĩa là cả phần.
- `NOTEBOOK_VERSION`: phiên bản code lõi mà cả nhóm thống nhất.
- `SAVE_CHECKPOINTS`: thường để `False`; chỉ cần checkpoint cho cấu hình cuối.

Không đổi `BASE_CONFIG`, split, model hoặc training loop giữa hai seed của cùng một cấu hình.

## 4. Dữ liệu và seed

- CIFAR-10 chính thức có 50.000 ảnh train và 10.000 ảnh test.
- Notebook dùng split seed 4653 để chia 45.000 train / 5.000 validation.
- Hai training seed chính thức là 42 và 2026.
- Train chỉ dùng augmentation khi chính cấu hình đó yêu cầu.
- Validation luôn dùng transform sạch.
- Test set chỉ được tạo trong cell cuối sau khi nhóm đã chọn cấu hình bằng validation.

Đây là các điều kiện để một phép so sánh có kiểm soát: cùng dữ liệu, cùng model nền, cùng seed và cùng ngân sách; chỉ đổi yếu tố đang khảo sát.

## 5. File đầu ra của mỗi lượt

Notebook ghi sau từng run để giảm rủi ro mất log khi Kaggle ngắt phiên:

```text
/kaggle/working/it4653/
├── summary_memberN_partX.csv
├── epoch_log_memberN_partX.csv
├── step_log_memberN_partX.csv
├── mean_std.csv
└── figures/
```

### `summary_*.csv`

Mỗi dòng là một lượt chạy `(experiment_id, seed)`, gồm cấu hình thực tế, best validation accuracy, best epoch, kết quả cuối, thời gian, GPU, phiên bản thư viện và notebook version. Đây là nhật ký chính theo yêu cầu PDF.

### `epoch_log_*.csv`

Mỗi dòng là một epoch, gồm train/validation loss, accuracy, learning rate và thời gian epoch. File này dùng cho đường cong theo epoch.

### `step_log_*.csv`

Ghi loss và learning rate mỗi 20 optimizer steps. File này dùng cho biểu đồ schedule theo bước mà đề tài 3 yêu cầu.

### `mean_std.csv` và `figures/`

Được sinh từ các file raw, không sửa tay. Đây là nguồn trực tiếp để làm bảng và hình trong báo cáo.

## 6. Chạy song song bằng ba tài khoản

Tất cả thành viên phải bắt đầu từ cùng một `NOTEBOOK_VERSION`.

| Thành viên | `MEMBER` | Phần chạy |
|---|---:|---|
| 1 | 1 | anchor, optimizer, normalization |
| 2 | 2 | learning-rate schedule |
| 3 | 3 | regularization |

Mỗi người:

1. chạy `DEBUG=True` một lần;
2. chuyển `DEBUG=False`;
3. chia `RUN_IDS` thành các part nếu cần;
4. Save Version sau mỗi part;
5. tải ba CSV về hoặc tạo private Kaggle Dataset từ output;
6. gửi CSV cho người tổng hợp.

Không gửi notebook đã sửa training loop riêng lẻ rồi tiếp tục chạy. Nếu phát hiện lỗi lõi, dừng, sửa bản chuẩn, tăng version và quyết định rõ các run cũ có phải chạy lại hay không.

## 7. Ghép log và tạo lại bảng/hình

1. Tạo một Kaggle Notebook tổng hợp từ notebook chuẩn.
2. Add Input toàn bộ CSV chính thức. Không Add Input pilot.
3. Đặt `DEBUG=False` và `NOTEBOOK_VERSION` đúng với official logs, rồi chạy cell import.
4. Chạy cell “Ghép CSV và vẽ tối thiểu 6 biểu đồ”.
5. Kiểm cột `seeds` trong `mean_std.csv` bằng 2 cho mọi cấu hình.
6. Download `mean_std.csv` và thư mục `figures`.
7. Đưa raw CSV vào `results/raw/`, bảng vào `results/mean_std.csv` và hình vào `results/figures/`.

Sáu hình mặc định:

1. optimizer train/validation loss theo epoch;
2. optimizer accuracy mean ± std;
3. schedule train loss theo global step;
4. schedule accuracy mean ± std;
5. normalization × batch size;
6. regularization accuracy mean ± std.

Nếu một số liệu trong report được cập nhật, hãy sinh lại bảng/hình từ log thay vì sửa trực tiếp file kết quả.

## 8. Final test

Chỉ sau khi chốt `FINAL_EXPERIMENT_ID` bằng validation:

```python
DEBUG = False
RUN_FINAL_TEST = True
FINAL_EXPERIMENT_ID = "id_da_chot"
```

Cell cuối train lại cấu hình đó với seed 42 và 2026, nạp best state rồi mới đánh giá test. Lưu `final_test.csv` vào repository. Không dùng test accuracy để quay lại đổi optimizer, LR hoặc regularization.

## 9. Khóa phiên bản trước khi nộp

Trong lần chạy chính thức, cell đầu in Python, PyTorch, torchvision và GPU. Sao chép đúng phiên bản package vào `requirements.txt`, ví dụ:

```text
torch==<phien-ban-da-in>
torchvision==<phien-ban-da-in>
numpy==<phien-ban-da-in>
pandas==<phien-ban-da-in>
matplotlib==<phien-ban-da-in>
```

Dấu `<...>` chỉ là minh họa; không nộp placeholder. Hãy dùng phiên bản thật của Kaggle session tạo ra kết quả chính. Ghi thêm `NOTEBOOK_VERSION`, link/commit GitHub và version Kaggle Dataset trong báo cáo hoặc README.

## 10. Kiểm tra như người chấm

Một thành viên không viết notebook nên làm thử trước hạn nộp:

1. mở link GitHub ở cửa sổ/thiết bị khác;
2. tải notebook mới hoàn toàn;
3. import vào Kaggle, bật GPU và Add Input;
4. chạy Mức A từ đầu tới cuối;
5. kiểm ba CSV xuất hiện;
6. mở một CSV và đối chiếu seed/config/kết quả;
7. Add Input raw logs chính thức và tạo lại `mean_std.csv` cùng sáu hình.

Nếu quy trình này thành công mà không cần hỏi người viết code, phần README/notebook đã đạt mục tiêu “chạy lại được”.
