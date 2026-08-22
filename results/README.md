# Quy ước kết quả

Thư mục này chứa **kết quả chạy thật** dùng trong báo cáo. Không commit dữ liệu CIFAR-10, checkpoint lớn hoặc CSV pilot.

Sau khi hoàn tất thí nghiệm, cấu trúc nên là:

```text
results/
├── raw/
│   ├── summary_member1_part*.csv
│   ├── summary_member2_part*.csv
│   ├── summary_member3_part*.csv
│   ├── epoch_log_member*.csv
│   └── step_log_member*.csv
├── mean_std.csv
└── figures/
    ├── 01_optimizer_loss.png
    ├── 02_optimizer_accuracy.png
    ├── 03_schedule_loss.png
    ├── 04_schedule_accuracy.png
    ├── 05_normalization.png
    └── 06_regularization.png
```

Quy tắc:

1. Mỗi dòng summary tương ứng đúng một `(experiment_id, seed)` và có `best_val_accuracy`, `test_loss`, `test_accuracy`.
2. Mỗi cấu hình chính thức có seed 42 và 2026.
3. Không sửa tay số liệu CSV; nếu có lỗi thì chạy lại và thay cả run.
4. `mean_std.csv` và hình phải được sinh từ raw CSV bằng notebook.
5. Số trong report/slide phải khớp các file ở đây.
6. Không trộn file `pilot*`, log notebook `v1` và log official test-all `v2`.

Git không lưu thư mục rỗng, vì vậy `raw/` và `figures/` chỉ xuất hiện sau khi nhóm thêm kết quả thật.
