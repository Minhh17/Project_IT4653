# Kế hoạch nhóm ba người

## Bảng phân công dùng khi nộp

Điền họ tên và MSSV thật trước khi nộp. Tỷ lệ dưới đây là gợi ý ban đầu và phải được điều chỉnh nếu khối lượng thực tế thay đổi.

| Thành viên | Họ tên - MSSV | Phần chịu trách nhiệm chính | Tỷ lệ dự kiến |
|---|---|---|---:|
| 1 | `[Điền họ tên - MSSV]` | ResNet, optimizer, normalization, anchor | 34% |
| 2 | `[Điền họ tên - MSSV]` | Scheduler, tích hợp notebook, ghép log và biểu đồ | 33% |
| 3 | `[Điền họ tên - MSSV]` | Data, augmentation, regularization, hỗ trợ final test | 33% |
| **Tổng** |  |  | **100%** |

Đây không chỉ là bảng hình thức: khi bảo vệ, mỗi người phải giải thích được phần mình nhận và các cell lõi dùng chung.

## Cách làm chung

- Một notebook chuẩn do Thành viên 2 tích hợp và đánh version `v1`, `v2`...
- Mỗi người dùng Copy & Edit đúng cùng version trên Kaggle.
- Mỗi người chỉ đổi `MEMBER`, `PART`, `RUN_IDS` và cấu hình thuộc nhánh mình.
- Thay đổi code lõi phải đưa lại cho người tích hợp; sau đó cả nhóm chuyển sang version mới.
- Cả ba phải đọc được các cell Data → Model → Optimizer → Train, không chỉ phần mình.

## Thành viên 1 - Optimizer, normalization và anchor

Phạm vi:

- Giải thích `BasicBlock`, shortcut và ResNet-18 CIFAR.
- Giải thích sáu optimizer, đặc biệt momentum, Nesterov, Adam và AdamW.
- Giải thích BN/LN/GN và ảnh hưởng của batch size.
- Chạy shared anchor.

Khối lượng: 14 cấu hình riêng × 2 seed = **28 runs**.

Bàn giao:

- `summary_member1_*.csv`, epoch/step logs.
- Hai hình optimizer và một hình normalization.
- Phần viết optimizer + normalization.

## Thành viên 2 - Schedule, ghép CSV và biểu đồ

Phạm vi:

- Giải thích công thức constant, step, cosine và warm-up.
- Điều phối pilot LR công bằng: cùng danh sách ứng viên/cùng ngân sách cho mỗi optimizer, chốt trước official run.
- Kiểm LR in trong log có đúng theo epoch.
- Add Input CSV của cả ba người, tạo `mean_std.csv` và sáu hình.
- Giữ notebook chuẩn và tăng `NOTEBOOK_VERSION` khi code lõi đổi.

Khối lượng: 5 cấu hình riêng × 2 seed = **10 runs**; constant/no-warm-up lấy từ anchor.

Bàn giao:

- `summary_member2_*.csv`, epoch/step logs.
- Hai hình schedule và bảng mean ± std chung.
- Phần viết schedule + thiết lập thí nghiệm.

## Thành viên 3 - Data và regularization

Phạm vi:

- Giải thích split 45k/5k và vì sao validation không augment.
- Giải thích crop/flip/jitter, Dropout, weight decay, early stopping.
- Phân tích riêng lẻ và cấu hình combined.
- Chỉ hỗ trợ final test sau khi nhóm đã chọn model bằng validation.

Khối lượng: 7 cấu hình riêng × 2 seed = **14 runs**; WD-only lấy từ anchor.

Bàn giao:

- `summary_member3_*.csv`, epoch/step logs.
- Hình regularization và phần viết tương ứng.

## Timeline gợi ý

### Ngày 1 - cả nhóm

1. Import notebook, Add Input CIFAR-10 và bật GPU.
2. Cùng đọc các cell theo thứ tự.
3. Mỗi người chạy `DEBUG=True`, tự xem batch/logits/loss và pilot 1 epoch.
4. Pilot cùng ngân sách để chốt LR từng optimizer; chốt baseline, 52 hay 54 runs và notebook version.

### Ngày 2–3 - chạy song song

1. Đặt `DEBUG=False`.
2. Chia `RUN_IDS` thành các part vừa thời lượng Kaggle.
3. Sau mỗi part, Save Version và giữ ba CSV.
4. Không đổi code/baseline giữa hai seed của cùng cấu hình.

### Ngày 4 - ghép và phân tích

1. Thành viên 2 Add Input toàn bộ CSV.
2. Chạy cell mean ± std và sáu biểu đồ.
3. Cả nhóm kiểm các config đủ hai seed.
4. Viết kết luận; nếu hai seed đảo xu hướng thì ghi “chưa đủ bằng chứng”.

### Ngày 5 - final test và bảo vệ

1. Chốt một cấu hình bằng validation.
2. Đổi `RUN_FINAL_TEST=True` đúng một lần cho cấu hình đó.
3. Mỗi người trình bày thử một phần không phải nhánh mình.
4. Nộp notebook version cuối, CSV, hình, report, slide và khai báo AI.
