# IT4653 — Đề tài 3: khảo sát kỹ thuật huấn luyện

Đây là base code PyTorch thuần cho khảo sát có kiểm soát trên ResNet-18/CIFAR-10. Thiết kế ưu tiên hai mục tiêu: **mỗi run có thể truy vết** và **cả ba thành viên giải thích được mọi dòng code khi bảo vệ**.

**Chạy lần đầu trên Kaggle:** đọc duy nhất [KAGGLE_START_HERE.md](docs/KAGGLE_START_HERE.md). Base config tự tìm CIFAR-10 đã gắn bằng Add Input; không tải dataset từ code.

> Trạng thái hiện tại: `draft-v1`. Các learning rate và cấu hình đối chứng là giá trị pilot đề xuất, chưa phải kết luận. Không chạy toàn bộ GPU queue trước khi hoàn tất checklist trong [EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md).

## Phạm vi đã dựng

- ResNet-18 cho ảnh 32×32: stem 3×3, stride 1, không max-pool.
- CIFAR-10 mặc định; split cố định 45.000 train / 5.000 validation; test bị tách khỏi train command.
- Sáu optimizer, ba scheduler có/không warm-up, ba normalization × ba batch size, chín cấu hình regularization.
- Hai seed `42`, `2026`; log theo step và epoch; best checkpoint theo validation accuracy.
- Queue riêng cho ba thành viên, tổng **27 cấu hình duy nhất / 54 seeded runs** sau khi dùng chung một anchor.
- Script tổng hợp mean ± sample standard deviation, audit log và tám biểu đồ.
- Fake-data smoke test, overfit-128 test, LR range test và cổng xác nhận trước khi đọc test set.

Chi tiết vì sao Kickoff ghi 52 nhưng danh sách chi tiết dẫn tới 54 runs nằm trong [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

## Cài đặt

Python 3.8+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Trên Kaggle, PyTorch thường đã có sẵn. Notebook mẫu kiểm tra phiên bản trước, sau đó cài project bằng `--no-deps` để không âm thầm thay CUDA/PyTorch giữa các máy. Dùng Add Input với bản CIFAR Python đã giải nén; `data.root: auto` tìm thư mục đó dưới `/kaggle/input`. Mọi official run phải có cùng phiên bản; phiên bản thực tế tự được ghi trong `environment.json`.

## Kiểm tra ngày đầu

Chạy lần lượt:

```bash
python scripts/check_configs.py
python -m unittest discover -s tests -v
python scripts/smoke_test.py
python scripts/overfit_128.py
```

`smoke_test.py` dùng tensor giả, CPU, một epoch và đi trọn pipeline. `overfit_128.py` dùng CIFAR-10 đã attach trên Kaggle và phải ghi nhớ được ít nhất 90% trên 128 ảnh; đây là kiểm tra lỗi code, không phải kết quả báo cáo.

Pilot một epoch thật:

```bash
python scripts/train.py \
  --config configs/base.yaml \
  --set experiment.id=pilot_one_epoch \
  --set experiment.label="Pilot one epoch" \
  --set train.epochs=1 \
  --set scheduler.warmup_epochs=0
```

Một run tạo ra:

```text
runs/<experiment_id>/seed_<seed>[_<attempt>]/
├── config.resolved.yaml
├── environment.json
├── status.json
├── metrics_step.csv
├── metrics_epoch.csv
├── summary.json
└── checkpoints/best.pt
```

Run directory là bất biến: code từ chối ghi đè. Nếu một phiên bị ngắt, đọc `status.json`, giữ log làm bằng chứng rồi chạy lại **cùng experiment id/seed** với `--set experiment.attempt=retry1`. Thư mục vật lý mới được tạo nhưng aggregator vẫn ghép đúng hai seed của cùng cấu hình.

Với queue, retry các thư mục hỏng bằng lệnh sau; thêm `--only <id>` nếu chỉ muốn một cấu hình:

```bash
python scripts/run_matrix.py configs/matrices/member1_optimizer_norm.yaml \
  --retry-failed --attempt retry1
```

## Quy trình thí nghiệm chính thức

1. Chạy LR range test/pilot với cùng ngân sách cho từng optimizer:

   ```bash
   python scripts/lr_range_test.py --tag sgd_pilot1 \
     --set optimizer.name=sgd --start-lr 1e-5 --end-lr 1
   ```

2. Chốt mọi ô còn mở trong protocol; cập nhật LR trong matrix; đổi `protocol_version`; chỉ sau review chéo mới đổi `approved: true`.

3. Kiểm tra queue mà chưa dùng GPU:

   ```bash
   python scripts/run_matrix.py configs/matrices/member1_optimizer_norm.yaml --dry-run
   python scripts/run_matrix.py configs/matrices/member2_schedules.yaml --dry-run
   python scripts/run_matrix.py configs/matrices/member3_regularization.yaml --dry-run
   ```

4. Mỗi người chạy queue của mình. Khi matrix còn là draft, pilot phải ghi rõ `--allow-draft`; official run không nên cần cờ này.

   ```bash
   python scripts/run_matrix.py configs/matrices/member1_optimizer_norm.yaml
   ```

5. Tổng hợp tự động và audit đủ seed. Ngoài mean±std, `paired_deltas.csv` giữ chênh lệch với anchor cho từng seed để nhìn ra trường hợp hai seed đảo dấu:

   ```bash
   python scripts/aggregate_results.py --strict
   python scripts/plot_results.py
   ```

   `runs/` và `results/` được ignore trong lúc chạy để metadata không tự báo dirty. Sau audit, dùng `git add -f` cho đúng các CSV/JSON/hình nhỏ cần nộp; không force-add checkpoint.

6. Chỉ sau khi đã khóa cấu hình cuối bằng validation: điền `configs/final_selection.yaml` với `experiment_id`, `semantic_fingerprint`, `training_git_commit` trong summary, protocol version, hai seed, người chọn và thời điểm; đặt `frozen: true` rồi commit file. Checkout đúng commit chứa quyết định này và đánh giá **cả hai seed** của đúng cấu hình đã chọn:

   ```bash
   python scripts/evaluate_test.py runs/<experiment_id>/seed_42 --confirm-final
   python scripts/evaluate_test.py runs/<experiment_id>/seed_2026 --confirm-final
   python scripts/aggregate_test_results.py
   ```

Không dùng test accuracy để đổi LR, epoch, augmentation hoặc chọn mô hình. Các ablation báo validation mean±std; test mean±std chỉ dành cho một cấu hình cuối đã freeze.

## Đọc code theo thứ tự

Để hiểu toàn bộ pipeline mà không bị nhảy lớp abstraction:

1. [`configs/base.yaml`](configs/base.yaml) — một run gồm những biến nào.
2. [`src/dlstudy/model.py`](src/dlstudy/model.py) — ResNet block, shortcut, BN/LN/GN, Dropout.
3. [`src/dlstudy/data.py`](src/dlstudy/data.py) — split, transform và DataLoader seed.
4. [`src/dlstudy/optimization.py`](src/dlstudy/optimization.py) — sáu optimizer và công thức LR theo step.
5. [`src/dlstudy/training.py`](src/dlstudy/training.py) — forward, loss, backward, update, validation, early stopping, checkpoint.
6. [`scripts/run_matrix.py`](scripts/run_matrix.py) — biến một matrix thành từng run độc lập.
7. [`scripts/aggregate_results.py`](scripts/aggregate_results.py) và [`scripts/plot_results.py`](scripts/plot_results.py) — số liệu báo cáo sinh từ log thế nào.

Bản walkthrough và câu hỏi tự kiểm tra nằm ở [CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md). Kế hoạch ba người nằm ở [TEAM_PLANS.md](docs/TEAM_PLANS.md); quy trình pin commit, chia session và ghép artifact từ ba tài khoản nằm ở [KAGGLE_WORKFLOW.md](docs/KAGGLE_WORKFLOW.md).

## Quy tắc so sánh

- Trong một nhánh chỉ thay đúng yếu tố đang khảo sát.
- LR của mỗi optimizer được tìm với cùng ngân sách, không ép cùng một LR.
- Batch-size study giữ LR cố định để không thêm biến “linear scaling”.
- Adam và AdamW được chạy với cùng weight decay dương; nếu weight decay bằng 0, hai thuật toán gần như trùng và phép so sánh mất ý nghĩa.
- Thời gian chỉ so giữa runs có cùng `gpu_name`; accuracy có thể tổng hợp trên hai máy nếu software/config/split trùng.
- Với hai seed, chỉ mô tả mean ± std. Nếu chênh lệch cùng cỡ std hoặc hai seed đảo thứ tự, kết luận là “chưa đủ bằng chứng”, không gọi đó là kiểm định ý nghĩa thống kê.

## Nguồn và phần nhóm tự viết

Kiến trúc dựa trên bài báo [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385); dữ liệu từ [trang CIFAR chính thức](https://www.cs.toronto.edu/~kriz/cifar.html). Cặp phiên bản PyTorch được chọn từ [bảng phiên bản chính thức](https://pytorch.org/get-started/previous-versions/); cách seed worker bám theo [hướng dẫn reproducibility của PyTorch](https://docs.pytorch.org/docs/stable/notes/randomness.html).

Toàn bộ phần ghép CIFAR ResNet, normalization wrapper, scheduler, training loop, logging, matrix runner, aggregator và plotting trong repository này là base code do nhóm cần đọc, kiểm thử và chịu trách nhiệm. Khi dùng thêm nguồn khác, ghi rõ file/phần kế thừa tại đây trước khi nộp.
