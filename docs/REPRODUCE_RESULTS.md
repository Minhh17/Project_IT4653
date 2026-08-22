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
RUN_ALL_CONFIGS = False
```

Chạy notebook từ trên xuống. Kết quả pilot chỉ để kiểm tra code, không đưa vào bảng báo cáo và không đọc test set.

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
RUN_ALL_CONFIGS = True
```

Chỉ cần `RUN_ALL_CONFIGS=True`; notebook tự đặt các giá trị còn lại cho full suite. Chế độ này gồm 26 cấu hình × 2 seed = 52 lượt và mỗi lượt tự đo test. Trên Kaggle T4, tổng thời gian gần như chắc vượt giới hạn 12 giờ của một Save & Run All; preset này phù hợp khi dùng GPU nhanh hơn hoặc môi trường không có giới hạn phiên. Muốn chia `RUN_IDS` trên T4, phải giữ `RUN_ALL_CONFIGS=False` rồi dùng chế độ member/part thông thường.

## 2. Chuẩn bị một Kaggle Notebook sạch

1. Tải `notebooks/DeTai3_Kaggle.ipynb` từ đúng release/commit GitHub mà nhóm ghi trong báo cáo.
2. Kaggle → **Create → New Notebook** → import file notebook.
3. Settings → Accelerator → chọn **GPU T4 x2**.
4. Add Input → thêm [CIFAR-10 Python](https://www.kaggle.com/datasets/pankrzysiu/cifar10-python), hoặc đúng dataset/version mà nhóm đã ghi trong báo cáo.
5. Dataset trên chứa `cifar-10-python.tar.gz`; notebook tự giải nén và kiểm `cifar-10-batches-py/data_batch_1` trước khi train.
6. Không cần chạy `pip install` vì các thư viện đã có trong Kaggle image. Nếu import báo thiếu thư viện, chỉ cài đúng package/phiên bản ghi trong `requirements.txt`.

Cell đầu sẽ dừng với thông báo rõ nếu không thấy GPU, nếu chọn P100, hoặc nếu input không có đúng một archive/thư mục CIFAR-10 Python.

## 3. Các giá trị được phép sửa

Trong lần chạy thông thường, chỉ sửa các giá trị ở đầu notebook:

```python
MEMBER = 1
DEBUG = True
PART = "pilot1"
RUN_IDS = []
RUN_ALL_CONFIGS = False
NOTEBOOK_VERSION = "v2"
SAVE_CHECKPOINTS = False
```

- `MEMBER`: chọn danh sách thí nghiệm của thành viên; `0` là tất cả.
- `DEBUG`: `True` chạy nhanh trên tập nhỏ; `False` chạy chính thức 20 epoch và hai seed.
- `PART`: tên phần để các phiên không ghi đè nhau.
- `RUN_IDS`: danh sách cấu hình cần chạy; rỗng nghĩa là cả phần.
- `RUN_ALL_CONFIGS`: `True` tự chạy toàn bộ 26 cấu hình × 2 seed; các nút member/debug/part/run IDs được preset tự ghi đè.
- `NOTEBOOK_VERSION`: phiên bản code lõi mà cả nhóm thống nhất.
- `SAVE_CHECKPOINTS`: thường để `False`; checkpoint tốt nhất được test ngay trong cùng run rồi giải phóng.

Không đổi `BASE_CONFIG`, split, model hoặc training loop giữa hai seed của cùng một cấu hình.

## 4. Dữ liệu và seed

- CIFAR-10 chính thức có 50.000 ảnh train và 10.000 ảnh test.
- Notebook dùng split seed 4653 để chia 45.000 train / 5.000 validation.
- Hai training seed chính thức là 42 và 2026.
- Train chỉ dùng augmentation khi chính cấu hình đó yêu cầu.
- Validation luôn dùng transform sạch.
- Validation chọn checkpoint có accuracy cao nhất trong các epoch đã chạy.
- Sau khi train kết thúc, notebook nạp checkpoint đó và đo test đúng một lần; test không tham gia backward, early stopping hay chọn checkpoint.
- Validation/test đều dùng transform sạch, `shuffle=False`; batch 8/32/128 chỉ là batch train, còn evaluation dùng batch 256 cố định.

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

Mỗi dòng là một lượt chạy `(experiment_id, seed)`, gồm cấu hình thực tế, best validation accuracy/epoch, test loss/accuracy, kết quả epoch cuối, thời gian train/test, GPU, phiên bản thư viện và notebook version. Đây là nhật ký chính theo yêu cầu PDF.

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
2. optimizer test accuracy mean ± std;
3. schedule train loss theo global step;
4. schedule test accuracy mean ± std;
5. normalization × batch size theo test accuracy;
6. regularization test accuracy mean ± std.

Nếu một số liệu trong report được cập nhật, hãy sinh lại bảng/hình từ log thay vì sửa trực tiếp file kết quả.

## 8. Protocol test-all

Trước official run, nhóm phải khóa 26 cấu hình, hai seed, LR và notebook `v2` trong GitHub. Với mỗi `(experiment_id, seed)`, notebook làm đúng một chuỗi:

```text
train 45k → validation 5k chọn best epoch → nạp best state → test 10k đúng một lần
```

Quy tắc checkpoint là maximum validation accuracy; nếu hai epoch bằng nhau, điều kiện `>` giữ epoch xuất hiện sớm hơn. Test dùng `model.eval()` và `torch.no_grad()`, không augmentation, không shuffle, batch eval 256 cố định. Không test thêm weights ở epoch cuối rồi chọn kết quả đẹp hơn.

Test-all được dùng để kiểm chứng mức tổng quát hóa **trong từng nhánh** so với anchor. Không xếp hạng toàn bộ 26 cấu hình như một cuộc thi vì mỗi nhánh thay đổi một câu hỏi khác nhau. Nếu test và validation cho xu hướng khác nhau, báo cả hai và thảo luận; không đổi LR/config rồi chạy lại có chọn lọc.

`RUN_ALL_CONFIGS=True` cung cấp đúng một preset chạy toàn bộ. Tuy nhiên [Kaggle yêu cầu Save & Run All hoàn tất trong 12 giờ](https://www.kaggle.com/docs/notebooks). Với tốc độ khoảng 15 phút/run batch 128 đã quan sát, riêng 40 run batch 128 đã gần 10 giờ; còn 12 run batch 8/32. Vì vậy trên T4 nên giữ `RUN_ALL_CONFIGS=False` và chia part hoặc ba tài khoản. Chỉ dùng preset một lần nếu GPU/môi trường đủ nhanh để hoàn tất dưới giới hạn phiên.

Các CSV `v1` cũ vẫn hữu ích để phân tích thử train/validation nhưng không có weights hoặc test metrics. Không trộn chúng vào bảng official `v2`; nếu chọn protocol test-all, chạy lại các ID đó bằng `v2`.

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
