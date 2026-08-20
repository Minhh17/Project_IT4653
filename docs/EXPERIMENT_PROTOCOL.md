# Experiment protocol — cần freeze trước official runs

Trạng thái: **DRAFT**  
Protocol version trong config: `draft-v1`

Mỗi thay đổi sau khi official runs bắt đầu phải tạo protocol version mới; không sửa YAML rồi trộn kết quả cũ/mới.

## Các quyết định đã đề xuất

| Hạng mục | Giá trị draft | Trạng thái |
|---|---|---|
| Dataset | CIFAR-10 | cần nhóm xác nhận |
| Split | 45.000 train / 5.000 val, seed 4653 | đề xuất khóa |
| Test | freeze một config trong `final_selection.yaml`, rồi test cả 2 seed | bắt buộc |
| Model | ResNet-18 CIFAR, stem 3×3/s1, no max-pool | bắt buộc |
| Epoch budget | max 20 | theo đề |
| Seeds | 42, 2026 | theo Kickoff |
| Loss | CrossEntropy, label smoothing 0 | đề xuất khóa |
| Checkpoint | max validation accuracy | đề xuất khóa |
| Shared anchor | SGD momentum, WD 5e-4, constant, BN, batch 128 | cần pilot + tài liệu |
| Warm-up | linear 2 epoch, factor 0.1 → 1.0 | cần pilot |
| Step decay | mỗi 7 epoch sau warm-up, gamma 0.1 | cần pilot |
| Cosine | theo optimizer step, min LR 1e-5 | cần pilot |
| Early stopping | val loss, patience 4, min_delta 0 | cần pilot |
| Dropout | 0.1/0.3/0.5 sau GAP | theo Kickoff + cần xác nhận vị trí |
| Augmentation | crop pad4 + flip p=.5 + jitter .1 | theo Kickoff + cần khóa tham số |
| LayerNorm | per-image C×H×W, affine per-channel | cần ghi đúng trong report |
| GroupNorm | tối đa 8 group, tự chọn ước số hợp lệ | cần khóa |
| Batch-size LR | giữ LR, không linear-scale | cần nhóm xác nhận |

## Protocol theo nhánh

### Optimizer

Giữ WD=5e-4, constant schedule, no warm-up, BN, batch 128, no augmentation/dropout/early stopping. Mỗi optimizer có cùng ngân sách LR range test, sau đó chốt LR riêng trước hai seed.

Matrix draft hiện gọi đúng biến thể **RMSProp với `alpha=0.99`, momentum 0.9**; không rút gọn tên thành RMSProp thuần trong bảng/hình. Đây vẫn là quyết định pilot phải được freeze cùng LR.

Lý do WD dương: với WD=0, `Adam` và `AdamW` cập nhật gần như giống nhau nên không khảo sát được khác biệt coupled/decoupled weight decay. Đây là một phần định nghĩa thuật toán trong nhánh optimizer; báo cáo phải nói rõ, không gọi đó là phép so sánh “không regularization”.

Trong code, WD chỉ áp dụng cho `weight` của Conv/Linear; bias và affine scale/bias của normalization nằm trong parameter group có WD=0. Quy tắc này giữ nguyên cho mọi optimizer.

Các LR `0.1/0.001` trong matrix chỉ là điểm bắt đầu pilot. Không đổi sau khi xem test.

### Schedule

Giữ toàn bộ shared anchor. Sáu cấu hình là constant/step/cosine × warm-up 0/2. Shared anchor cung cấp constant/no-warm-up; thành viên 2 chỉ chạy năm cấu hình còn lại.

Scheduler được cập nhật theo optimizer step. So convergence bằng global step và epoch; batch size của nhánh này cố định.

### Normalization

Giữ shared anchor ngoài normalization và batch size. Chín ô BN/LN/GN × 8/32/128. Không scale LR theo batch vì như vậy sẽ thay thêm một yếu tố. Batch size đồng thời làm thay đổi số optimizer updates/epoch và cumulative WD, nên không diễn giải riêng hiệu số BN(b8)−BN(b128) là tác động của batch statistics. So BN với LN/GN **trong cùng batch size**, rồi xem chênh lệch tương đối đó thay đổi ra sao qua 8/32/128; chỉ xu hướng tương tác này mới hỗ trợ một giải thích riêng cho BN.

### Regularization

Chín cấu hình: không dùng regularizer đang khảo sát; WD; Dropout .1/.3/.5; augmentation; early stopping; WD+augmentation; WD+Dropout .3+augmentation+early stopping. Shared anchor là WD-only. Baseline đầu vẫn có BatchNorm, vì vậy báo cáo không gọi nó là “hoàn toàn không regularization”.

Early stopping có `max_epochs=20`; dừng sớm chính là treatment nên số update/thời gian có thể thấp hơn. Accuracy, `epochs_completed` và `training_seconds` cần báo song song. Không so trực tiếp `mean_val_accuracy_over_epochs` của run dừng sớm với run đủ 20 epoch vì horizon khác nhau.

## Freeze record

Điền trước official run:

- Ngày/giờ freeze:
- Git commit:
- Người duyệt 1:
- Người duyệt 2:
- Dataset/split checksum pilot:
- GPU dự kiến:
- LR cuối của SGD:
- LR cuối của SGD momentum:
- LR cuối của Nesterov:
- LR cuối của RMSProp:
- LR cuối của Adam:
- LR cuối của AdamW:
- Xác nhận số regularization configs từ giảng viên:
- Protocol version mới:

`train.deterministic=true` bật deterministic algorithms với `warn_only=True`. Nếu console xuất cảnh báo operation không deterministic, run không được tuyên bố là tái lập bit-exact; nhóm phải lưu cảnh báo, điều tra và chạy lại reproducibility gate trước khi freeze.

Sau khi điền, cập nhật `configs/base.yaml`, ba matrix, chạy `scripts/check_configs.py`, review diff rồi mới đặt `approved: true`.
