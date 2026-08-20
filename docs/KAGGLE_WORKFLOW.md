# Workflow Kaggle cho ba tài khoản

Nếu đây là lần đầu, đọc bản ngắn [KAGGLE_START_HERE.md](KAGGLE_START_HERE.md) trước. Tài liệu này chỉ chứa phần vận hành chi tiết hơn.

## 1. Chuẩn bị một nguồn code duy nhất

Workspace hiện chưa có Git history. Trước pilot chính thức, một thành viên phải tạo Git repository, commit sạch, push lên GitHub và ghi SHA đã duyệt vào freeze record. Không official-run khi `git rev-parse HEAD` chưa trả về commit hoặc `git status --short` còn nội dung.

Mỗi notebook:

1. Clone đúng repository.
2. `git checkout <COMMIT_SHA>` thay vì chạy theo đầu branch thay đổi.
3. In và đối chiếu Python/PyTorch/torchvision/GPU.
4. Assert CUDA có sẵn.
5. Add Input cùng bản CIFAR Python đã giải nén và chạy `scripts/kaggle_check.py`.
6. Chạy config checker và smoke/pilot mà nhóm muốn tự kiểm chứng.

Ba tài khoản phải dùng cùng commit/protocol/software. Nếu Kaggle image khác phiên bản giữa thời điểm chạy, dừng và thống nhất lại; không trộn silently.

## 2. Chia queue thành chunk

Không nhất thiết để một cell chạy cả queue tới khi session timeout. Liệt kê id:

```bash
python scripts/run_matrix.py configs/matrices/member1_optimizer_norm.yaml --dry-run
```

Chạy từng config, hai seed của nó, bằng `--only`:

```bash
python scripts/run_matrix.py configs/matrices/member1_optimizer_norm.yaml \
  --only opt_sgd
```

Sau mỗi config, kiểm hai `summary.json`. Khi session hỏng:

```bash
python scripts/run_matrix.py configs/matrices/member1_optimizer_norm.yaml \
  --only opt_sgd --retry-failed --attempt retry1
```

Sau retry, gọi queue bình thường để chạy các config còn thiếu. Không xóa hay nối CSV hỏng.

## 3. Không mất artifact khi session kết thúc

Cuối mỗi session, đóng gói log nhỏ (không checkpoint):

```bash
python scripts/export_logs.py \
  --output /kaggle/working/member1_logs_part1.tar.gz
```

Save Notebook Version với output hoặc tải cả `.tar.gz` và `.sha256`. Checkpoint của mọi run giữ trong Kaggle output/storage của owner cho tới khi nhóm chọn final config; không commit Git. Sau khi chọn, owner export riêng hai seed của final config với `--only-experiment <id> --include-checkpoints`.

`import_logs.py` tự kiểm companion `.sha256` nếu hai file nằm cạnh nhau; thiếu checksum sẽ hiện warning, sai checksum thì dừng trước khi giải nén.

## 4. Ghép ba tài khoản

Người 2 giữ vai trò integrator. Tải/attach tất cả archive vào một Kaggle session hoặc máy local rồi chạy:

```bash
python scripts/import_logs.py \
  /path/member1_logs.tar.gz \
  /path/member2_logs.tar.gz \
  /path/member3_logs.tar.gz
python scripts/aggregate_results.py --strict
python scripts/plot_results.py
```

Importer từ chối ghi đè nếu cùng path có nội dung khác. Shared anchor chỉ nằm trong archive của thành viên 1 nhưng nhờ `comparison_groups` sẽ xuất hiện trong đủ bốn bảng/nhánh sau merge.

## 5. Final test

Integrator điền `training_git_commit` cùng fingerprint trong `configs/final_selection.yaml` từ validation summary, đặt `frozen: true` rồi commit quyết định đó trước khi lấy checkpoint. Cả hai lượt test phải checkout **cùng commit chứa final selection**, sạch Git, cùng CUDA/software/GPU; `aggregate_test_results.py` sẽ audit các trường này. Sau đó import hai checkpoint được chọn, test seed 42/2026 và aggregate. Không attach checkpoint của cấu hình khác vào bước này để tránh test leakage.
