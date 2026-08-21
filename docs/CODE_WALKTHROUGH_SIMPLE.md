# Đọc và hiểu notebook theo từng phần

Tài liệu này giải thích bản chất code trong `notebooks/DeTai3_Kaggle.ipynb`. Hãy mở notebook cạnh tài liệu và đọc từ cell 1 tới cell 10. Mục tiêu không phải học thuộc cú pháp, mà trả lời được: dữ liệu đi đâu, tham số nào thay đổi, gradient được tạo và model được cập nhật ở chỗ nào.

## Bức tranh toàn bộ

```text
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
Validation → lưu best result → CSV → mean ± std → biểu đồ
```

## Cell 1 - import và công tắc chạy

- `MEMBER` chọn danh sách thí nghiệm, không thay đổi thuật toán train.
- `DEBUG=True` giảm dữ liệu, epoch và seed để nhóm kiểm tra nhanh.
- `PART` chỉ là tên file, giúp hai phiên không ghi đè nhau.
- `RUN_IDS` lọc đúng cấu hình cần chạy.
- `NOTEBOOK_VERSION` cho biết ba người có đang dùng cùng code lõi không.
- `SAVE_CHECKPOINTS=False` vì ablation cần số liệu, không cần lưu 52 model lớn.

`torch.cuda.is_available()` kiểm GPU thật của Kaggle. Notebook từ chối P100 để tránh lỗi tương thích PyTorch hiện tại và cố ý dùng `cuda:0` của T4 x2. Nếu Input chỉ có `cifar-10-python.tar.gz`, module chuẩn `tarfile` giải nén archive vào vùng ghi được `/kaggle/working`; `DATA_ROOT` sau đó là thư mục cha của `cifar-10-batches-py` mà `torchvision.datasets.CIFAR10` cần.

## Cell 2 - baseline và thí nghiệm

`BASE_CONFIG` chứa mọi giá trị mặc định của một run. Một experiment chỉ chứa phần thay đổi:

```python
config = {**BASE_CONFIG, **experiment_config, "epochs": EPOCHS}
```

Python ghép dictionary từ trái sang phải. Giá trị trong `experiment_config` ghi đè baseline. Ví dụ:

```python
experiment("opt_adam", "Adam", "optimizer", optimizer="adam", lr=0.001)
```

chỉ đổi optimizer và LR đã chọn cho optimizer đó; batch size, normalization, schedule, epoch và các yếu tố khác giữ nguyên. Vì LR cũng đổi, kết luận chính xác là “so sánh optimizer dưới một quy trình chọn LR công bằng”, không phải ảnh hưởng thuần túy của tên optimizer. Trước official run, mỗi optimizer phải được pilot với cùng số ứng viên LR và cùng ngân sách.

`ANCHOR` là cấu hình đối chứng chung. Chạy anchor một lần nhưng có thể dùng làm mốc trong bốn nhánh; không cần train lại cùng một cấu hình chỉ vì nó xuất hiện ở nhiều biểu đồ.

## Cell 3 - dữ liệu

Ba dataset object cùng trỏ tới CIFAR-10 train nhưng dùng transform khác nhau:

- `TRAIN_CLEAN`: ảnh train không augmentation;
- `TRAIN_AUGMENTED`: crop, flip và color jitter;
- `EVAL_DATA`: ảnh sạch dùng cho validation.

`torch.randperm(..., generator=split_generator)` tạo một hoán vị cố định bằng seed 4653. 45.000 chỉ số đầu là train, 5.000 chỉ số sau là validation. Training seed 42/2026 không làm thay đổi split này.

`DataLoader` gom ảnh thành mini-batch. Train dùng `shuffle=True`; validation dùng `shuffle=False`. Generator của train được seed để thứ tự batch có thể tái lập.

## Cell 4 - ResNet-18 cho CIFAR-10

Một `BasicBlock` có hai convolution 3×3. Kết quả nhánh chính được cộng với `residual`:

```python
return self.relu(outputs + residual)
```

Đường tắt giúp gradient đi qua mạng sâu dễ hơn. Nếu số channel hoặc kích thước không gian thay đổi, convolution 1×1 trong `shortcut` đưa residual về cùng shape trước khi cộng.

ResNet-18 có bốn stage, mỗi stage hai block: tổng cộng 8 block × 2 convolution = 16 convolution, cộng stem và classifier tạo cách gọi “18 lớp”.

Khác ResNet cho ImageNet, ảnh CIFAR chỉ 32×32 nên stem dùng kernel 3×3, stride 1 và không max-pool đầu mạng.

Ba normalization:

- BatchNorm dùng thống kê theo batch và có running statistics;
- `LayerNorm2d` đổi NCHW sang NHWC rồi dùng `nn.LayerNorm(C)` tại từng vị trí H×W;
- GroupNorm chia channel thành 8 nhóm và không phụ thuộc các ảnh khác trong batch.

Khi batch size đổi, số optimizer step trong 20 epoch cũng đổi. Vì vậy hãy so BN với LN/GN **ở cùng batch size** trước; chỉ khi BN giảm tương đối mạnh hơn hai phương pháp kia ở batch nhỏ mới có bằng chứng cho ảnh hưởng của batch statistics.

Dropout nằm sau global average pooling và trước classifier. Khi train, một tỷ lệ feature được đặt về 0; khi `model.eval()`, Dropout tự tắt.

## Cell 5 - optimizer và learning rate

`build_optimizer` dùng `if/elif` để map tên cấu hình sang optimizer PyTorch. Trước đó, code chia tham số thành hai nhóm: ma trận weight của Conv/Linear (`ndim > 1`) nhận weight decay; bias và tham số normalization (`ndim == 1`) không nhận decay. Hai nhóm vẫn chứa toàn bộ tham số cần cập nhật.

`learning_rate_for_epoch` trả LR cho mỗi epoch:

- constant: giữ nguyên LR;
- step: nhân `gamma` sau mỗi `step_size` epoch;
- cosine: giảm mượt theo nửa đường cosine;
- warm-up: tăng tuyến tính từ LR nhỏ tới base LR trong vài epoch đầu.

`set_learning_rate` cập nhật LR trong từng parameter group của optimizer trước khi train epoch đó.

## Cell 6 - train và validation

Trình tự quan trọng của một mini-batch:

```python
optimizer.zero_grad()
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

1. `zero_grad`: xóa gradient của batch trước;
2. forward: model tạo logits;
3. loss: CrossEntropy đo sai khác với nhãn thật;
4. backward: autograd tính gradient cho từng tham số;
5. step: optimizer dùng gradient để cập nhật tham số.

`model.train()` bật hành vi train của Dropout/BatchNorm. `model.eval()` chuyển chúng sang hành vi đánh giá. Decorator `@torch.no_grad()` giúp validation không giữ đồ thị gradient, tiết kiệm bộ nhớ.

`best_state` là bản weights tại epoch có validation accuracy cao nhất. Early stopping theo dõi validation loss và dừng khi không cải thiện đủ số epoch `patience`.

Cuối run, hàm trả bốn thứ:

- một dòng summary;
- log từng epoch;
- log thưa từng step;
- best model state trong bộ nhớ.

## Cell 7 - nhóm tự kiểm tra

Cell lấy một batch thật, dựng model thật, chạy forward và backward thật trên GPU. Ba điều nên nhìn:

- input có shape `[batch, 3, 32, 32]`;
- logits có shape `[batch, 10]`;
- loss là số hữu hạn, không phải `nan`.

Sau đó chạy pilot một epoch và xem loss/accuracy. Đây là kiểm tra trực tiếp phục vụ học và debug, không phải kết quả báo cáo.

## Cell 8 - vòng chạy và CSV

Vòng ngoài đi qua cấu hình, vòng trong đi qua hai seed. Sau mỗi run, notebook nối metadata vào history và ghi CSV ngay; nếu Kaggle ngắt sau run thứ ba thì ba run đầu vẫn còn log.

Tên file chứa `MEMBER` và `PART`, nên nhóm biết file đến từ ai và phiên nào. `SAVE_CHECKPOINTS` chỉ bật cho model thật sự cần giữ.

## Cell 9 - tổng hợp và biểu đồ

`read_matching_csv` đọc file trong `/kaggle/working` và các Input được attach. `drop_duplicates(["experiment_id", "seed"])` giữ một dòng cho mỗi run chính thức.

`groupby(...).agg(...)` gom hai seed của cùng cấu hình để tính:

- `accuracy_mean`: trung bình;
- `accuracy_std`: sample standard deviation;
- `seeds`: số seed thực có.

Error bar trong biểu đồ là độ lệch chuẩn giữa hai seed, không phải confidence interval hay kiểm định thống kê.

## Cell 10 - final test

Trong giai đoạn chọn cấu hình, test set chưa được tạo. Sau khi nhóm chốt ID tốt nhất bằng validation, cell cuối train lại đúng cấu hình với hai seed, nạp best weights rồi mới đo test.

Nếu nhìn test rồi quay lại đổi LR/model, test đã trở thành validation và con số cuối không còn là đánh giá khách quan.

## Câu hỏi cả ba người nên tự trả lời trước bảo vệ

1. Vì sao CIFAR ResNet không dùng conv 7×7 và max-pool đầu mạng?
2. Shortcut giải quyết vấn đề gì và khi nào cần convolution 1×1?
3. Tại sao phải gọi `zero_grad` trước `backward`?
4. `train()` và `eval()` khác nhau ở BatchNorm/Dropout thế nào?
5. Tại sao validation không dùng augmentation ngẫu nhiên?
6. Tại sao split seed tách khỏi training seed?
7. Tại sao một ablation chỉ được đổi một yếu tố?
8. Warm-up, step decay và cosine thay LR theo công thức nào?
9. Adam và AdamW khác nhau ở cách xử lý weight decay thế nào?
10. Mean ± std của hai seed cho phép kết luận tới mức nào?
11. Vì sao không được dùng test để chọn cấu hình?
12. Một dòng trong `summary_*.csv` được tạo từ đoạn code nào?

Mỗi người nên mở notebook và lần theo câu trả lời tới đúng cell, thay vì học thuộc tài liệu này.
