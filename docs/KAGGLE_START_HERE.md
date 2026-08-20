# Bắt đầu từ số 0 trên Kaggle

Đây là luồng ngắn nhất. GitHub giữ **code**, Kaggle giữ **dataset, GPU và log chạy**.

| Nơi | Đưa gì vào |
|---|---|
| GitHub nhóm | `src/`, `scripts/`, `configs/`, `tests/`, `docs/`, `notebooks/`, README và requirements |
| Không đưa lên GitHub | CIFAR-10, `.venv`, `runs/`, checkpoint, archive lớn |
| Kaggle Input | CIFAR-10 bản Python đã giải nén; về sau thêm archive log của ba người |
| Kaggle Working | repo clone, log và checkpoint của session hiện tại |

## 1. Một người tạo GitHub repo

Repo public là đơn giản nhất. Nếu dùng private repo, lưu token bằng Kaggle Secrets; tuyệt đối không viết token vào notebook.

```bash
git init
git add .gitignore README.md CONTRIBUTING.md pyproject.toml requirements.txt \
  requirements-dev.txt configs src scripts tests docs notebooks
git commit -m "Initial experiment base"
git branch -M main
git remote add origin https://github.com/<nhom>/<repo>.git
git push -u origin main
git rev-parse HEAD
```

Ba người clone repo để đọc/sửa code, mỗi người làm branch riêng rồi Pull Request vào `main`. Trước khi chạy, cả nhóm chốt **một full commit SHA**; không chạy theo `main` đang thay đổi.

## 2. Mỗi người tạo một Kaggle Notebook

1. Kaggle → **Create → New Notebook**.
2. Settings → **Accelerator → GPU**; bật Internet để clone GitHub.
3. Input → **Add Input** → tìm CIFAR-10 Python.
4. Chỉ chọn bản có thư mục `cifar-10-batches-py` với `data_batch_1`…`data_batch_5`, `test_batch`, `batches.meta`. Không chọn bản CSV/PNG hay chỉ có `.tar.gz`.
5. Cả ba người gắn cùng một dataset/version.
6. Upload/import [kaggle_runner.ipynb](../notebooks/kaggle_runner.ipynb) vào Kaggle.

Trong notebook, điền `REPO_URL` và full `COMMIT_SHA`, rồi Run All tới cell preflight. Code tự tìm đúng một `/kaggle/input/**/cifar-10-batches-py`; không cần chép dataset và không cần sửa path trong YAML.

## 3. Chạy một chunk

Mỗi `--only` chạy luôn hai seed `42` và `2026` của một cấu hình:

```bash
python scripts/run_matrix.py <matrix-cua-minh> --dry-run
python scripts/run_matrix.py <matrix-cua-minh> --only <experiment-id>
```

- Thành viên 1: `configs/matrices/member1_optimizer_norm.yaml`.
- Thành viên 2: `configs/matrices/member2_schedules.yaml`.
- Thành viên 3: `configs/matrices/member3_regularization.yaml`.

Hiện matrix còn `approved: false`: pilot thêm `--allow-draft`. Official run chỉ bắt đầu sau khi nhóm freeze protocol, đặt cả ba matrix thành `approved: true`, commit và checkout SHA mới.

## 4. Cuối mỗi Kaggle session

```bash
python scripts/export_logs.py \
  --output /kaggle/working/member1_part1.tar.gz
```

Save Notebook Version hoặc tải cả `.tar.gz` và `.sha256`. Không dựa vào session đang mở để giữ dữ liệu. Checkpoint giữ trong output của người chạy; chỉ export checkpoint của cấu hình cuối đã chọn.

## 5. Người 2 ghép kết quả

Tạo một Kaggle Notebook tích hợp, Add Input output/archive của cả ba người, clone cùng code commit, rồi:

```bash
python scripts/import_logs.py \
  /kaggle/input/<member1-output>/member1_part1.tar.gz \
  /kaggle/input/<member2-output>/member2_part1.tar.gz \
  /kaggle/input/<member3-output>/member3_part1.tar.gz
python scripts/aggregate_results.py --strict
python scripts/plot_results.py
```

Sau audit, nhóm có thể push các CSV/JSON/hình nhỏ đã duyệt lên GitHub. Không push dataset hoặc checkpoint.

Quy trình retry và final test chi tiết nằm trong [KAGGLE_WORKFLOW.md](KAGGLE_WORKFLOW.md).
