# Lộ trình đọc và hiểu toàn bộ base code

Mục tiêu không phải học thuộc syntax. Với mỗi file, một thành viên phải trả lời được: dữ liệu vào là gì, đầu ra là gì, state nào thay đổi, và sai ở đây sẽ làm kết luận khoa học lệch thế nào.

## Buổi đọc code 90 phút

### 0–10 phút: config là hợp đồng

Đọc `configs/base.yaml`, rồi `config.py` và `utils.py`.

- YAML chứa toàn bộ yếu tố có thể ảnh hưởng run.
- Matrix chỉ override vài dotted keys; phần còn lại kế thừa base.
- `validate_config()` dừng trước khi dùng GPU.
- `semantic_fingerprint()` bỏ seed và tên run để nhận biết hai config khoa học giống nhau.
- `utils.py` gom seed, chọn device, metadata Git/software và cách ghi JSON/YAML nguyên tử; không có training logic ẩn trong helper.

Tự thử: đổi `model.dropout=1.0`, chạy checker và giải thích vì sao bị chặn.

### 10–25 phút: dữ liệu và leakage

Đọc `splits.py`, rồi `data.py`.

- `random.Random(split_seed).shuffle` tạo một permutation cố định.
- Hai object CIFAR dùng cùng ảnh nhưng transform khác: train có thể augment, validation luôn deterministic.
- DataLoader train/val dùng generator riêng để việc tạo worker validation không thay thứ tự shuffle của epoch sau.
- `include_test=False` trong training là firewall, không chỉ là quy ước bằng lời.

Tự thử: chạy unit test split; đổi split seed và nhìn checksum đổi.

### 25–45 phút: ResNet-18 CIFAR

Đọc `model.py` từ `BasicBlock.forward()` ra ngoài.

- Main path học residual `F(x)` qua hai convolution.
- Shortcut mang `x`; khi kích thước đổi, conv 1×1 biến nó thành cùng shape.
- Output là `ReLU(F(x)+shortcut(x))`.
- Stage 2/3/4 stride 2 nên spatial size giảm một nửa.
- Adaptive average pool biến mỗi channel thành một số; Dropout đặt trước classifier.

Ba normalization:

- BatchNorm: mean/variance có batch dimension, có running statistics.
- LayerNorm2d của repo: mỗi ảnh dùng toàn bộ C×H×W của feature map; không phụ thuộc ảnh khác trong batch.
- GroupNorm: chia channel thành group; statistics theo từng ảnh/từng group.

Tự thử: dùng base channels 8, in shape `[2,3,32,32]`, out shape phải `[2,10]`.

### 45–60 phút: optimizer và scheduler

Đọc `optimization.py`.

- `build_optimizer` là ánh xạ tên → lớp/arguments PyTorch hiển nhiên.
- SGD không momentum; SGD momentum có velocity; Nesterov nhìn trước; RMSProp chia gradient theo moving square; Adam có first/second moments; AdamW tách decay khỏi adaptive update.
- `LearningRateSchedule.factor(step)` trả hệ số nhân base LR.
- Warm-up chạy trước schedule chính; constant=1; step nhân gamma; cosine đi trơn tới min LR.

Tự thử trên giấy: 2 epoch warm-up, 100 step/epoch; tính factor tại step 0, 199, 200.

### 60–80 phút: một epoch huấn luyện

Đọc `training.py` theo thứ tự gọi, bắt đầu từ `run_training()` rồi vào `train_one_epoch()`.

Với mỗi batch:

1. Chuyển tensor sang device.
2. Đặt LR của đúng global step.
3. Xóa gradient cũ.
4. Forward tạo logits.
5. Cross-entropy so logits với class target.
6. Backward tính gradient.
7. Optimizer cập nhật parameter.
8. Log loss/accuracy/LR.

Validation có `torch.no_grad()` và `model.eval()`: không lưu graph, Dropout tắt, BatchNorm dùng running statistics. Checkpoint chọn theo validation accuracy; early stop có monitor riêng và chỉ bật ở treatment tương ứng.

Tự thử: giải thích vì sao quên `zero_grad()` làm gradient cộng dồn; vì sao không dùng train accuracy để chọn checkpoint.

### 80–90 phút: từ queue tới báo cáo

- `train.py` chỉ ghép base config + override rồi gọi `run_training()`.
- `run_matrix.py` tạo command cho từng config × seed, kiểm fingerprint trước khi bỏ qua run completed và tạo attempt mới khi retry.
- `aggregate_results.py` đối chiếu đủ matrix/seed/split/commit rồi tính mean, sample SD và paired delta.
- `plot_results.py` chỉ đọc log đã audit; không chứa hard-coded accuracy.
- `evaluate_test.py` là đường duy nhất tạo test loader, kiểm frozen selection/checkpoint/seed rồi lưu môi trường test.
- `aggregate_test_results.py` chỉ ghép đúng hai seed của final selection khi code/software/hardware giống nhau.

Các CLI hỗ trợ cũng phải đọc, không coi là “code phụ”:

- `check_configs.py`: chứng minh matrix resolve hợp lệ và đếm đúng 27 config/54 run.
- `smoke_test.py`: đi trọn pipeline bằng dữ liệu giả; `overfit_128.py`: kiểm model có thể ghi nhớ tập nhỏ.
- `lr_range_test.py`: tăng LR có kiểm soát và lưu provenance của pilot.
- `export_logs.py`/`import_logs.py`: đóng gói và ghép artifact ba tài khoản mà không ghi đè xung đột.
- Năm file trong `tests/`: mỗi test mô tả một invariant; đọc cả điều kiện skip để không nhầm “skip” với “pass”.
- `__init__.py`: khai báo package/version, không có side effect.

Sau buổi 90 phút, chia ba phiên 45 phút theo bảng owner trong `TEAM_PLANS.md` để đọc từng dòng các script còn lại. Một file chỉ được ký “đã hiểu” khi owner và reviewer đều mô tả được nhánh lỗi của nó.

## Cách review từng dòng

Mỗi file dùng quy trình ba lượt:

1. Owner giải thích từng function mà không chạy code.
2. Reviewer đưa một thay đổi giả (batch=8, warmup=2, dropout=.3) và owner lần theo dữ liệu/state.
3. Cả hai đặt một bug có chủ ý trong bản nháp, dự đoán test/log nào bắt được, rồi hoàn nguyên trước commit.

Không giữ dòng code mà không ai trong nhóm trả lời được “tại sao cần nó”. Nếu một tối ưu hiệu năng làm code khó giải thích, chỉ thêm sau khi baseline đã đúng và đo được lợi ích.

## 15 câu hỏi chung trước bảo vệ

1. Vì sao stem ImageNet không phù hợp ảnh 32×32?
2. Residual connection giúp gradient thế nào?
3. Seed model và split seed khác vai trò gì?
4. Vì sao validation không augment?
5. `model.train()` và `model.eval()` thay đổi module nào?
6. Log loss theo step khác loss theo epoch ở đâu?
7. Vì sao mỗi optimizer cần LR riêng nhưng cùng LR-search budget?
8. Warm-up được ghép với cosine/step theo công thức nào?
9. Vì sao AdamW với WD=0 gần Adam?
10. BN batch nhỏ có variance estimate nhiễu hơn như thế nào?
11. LayerNorm2d của repo có đúng như `nn.LayerNorm(C,H,W)` hoàn toàn không? (Không: statistics tương ứng nhưng affine chỉ per-channel; phải nêu rõ.)
12. Early stopping chọn epoch dựa trên train hay validation?
13. Best checkpoint và final epoch khác nhau ra sao?
14. Với hai seed, khi nào nhóm nói “chưa đủ bằng chứng”? 
15. Vì sao test command tách hẳn training command?
