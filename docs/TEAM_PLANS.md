# Ba kế hoạch chi tiết cho ba thành viên

Thay `Thành viên 1/2/3` bằng họ tên và MSSV trước khi nộp. Tỉ lệ đóng góp ban đầu có thể để 34/33/33, nhưng phải cập nhật theo commit, run log, hình và phần viết thực tế; tổng cuối cùng đúng 100%.

## Phần chung bắt buộc — cả ba người, ngày 1

| Thời điểm | Việc | Bằng chứng hoàn thành |
|---|---|---|
| Sáng | Cùng đọc PDF, Kickoff, README và protocol | trả lời thống nhất phạm vi/số run |
| Sáng | Mỗi máy clone cùng commit, ghi Python/torch/torchvision/GPU | ba `environment.json` pilot |
| Trưa | `check_configs`, unit tests, fake smoke | terminal log hoặc screenshot |
| Chiều | overfit 128 ảnh, pilot 1 epoch CIFAR | curve hợp lý, train accuracy ≥90% ở overfit |
| Chiều | chạy lặp cùng seed trên cùng máy | so config/split/log; ghi sai số nếu có |
| Cuối ngày | freeze protocol và review ba matrix | hai reviewer ký freeze record |

Quy ước tích hợp:

- Một người không tự sửa file thuộc nhánh người khác rồi chạy official; mở PR để owner review.
- Mỗi PR code cần một reviewer; mỗi thay đổi config khoa học cần hai reviewer.
- Không commit dataset/checkpoint. Commit config, summary, metrics CSV và figures cần báo cáo.
- Mỗi run chính thức phải cùng protocol version và commit. Dirty git status phải được giải thích.
- Khi một run hỏng, không sửa CSV; giữ `status.json`, chạy cùng id/seed với `experiment.attempt=retry1`.
- Với queue, dùng `run_matrix.py <matrix> --retry-failed --attempt retry1`; nếu retry cũng hỏng, tăng thành `retry2`.

### Owner/reviewer cho toàn bộ code

Owner là người giải thích dòng code đầu tiên và duyệt thay đổi, **không phải** người duy nhất cần hiểu file. Trước bảo vệ, cả ba người vẫn phải đọc toàn bộ theo `CODE_WALKTHROUGH.md`; reviewer phải lần được input → state → output mà không nhờ owner.

| File/phần | Owner | Reviewer chính |
|---|---|---|
| `base.yaml`, `final_selection.yaml`, `config.py`, `train.py`, `check_configs.py`, `run_matrix.py` | Thành viên 2 | Thành viên 1 |
| `splits.py`, `data.py` | Thành viên 3 | Thành viên 1 |
| `model.py` | Thành viên 1 | Thành viên 3 |
| `optimization.py`: optimizer/parameter groups | Thành viên 1 | Thành viên 2 |
| `optimization.py`: LR schedule | Thành viên 2 | Thành viên 1 |
| `training.py`: loop/checkpoint/log | Thành viên 1 | Thành viên 2 và 3 |
| `training.py`: early stopping/clean train evaluation | Thành viên 3 | Thành viên 1 |
| `utils.py`, `__init__.py`, `export_logs.py`, `import_logs.py` | Thành viên 2 | Thành viên 3 |
| `lr_range_test.py`, `aggregate_results.py`, `plot_results.py` | Thành viên 2 | Thành viên 1 |
| `evaluate_test.py`, `aggregate_test_results.py` | Thành viên 3 | Thành viên 2 |
| `smoke_test.py`, `overfit_128.py`, toàn bộ `tests/` | Thành viên 3 | Thành viên 1 |
| Notebook Kaggle và tài liệu quy trình | Thành viên 2 ghép | Thành viên 1 và 3 |

Mỗi người sở hữu matrix của nhánh mình. Một PR chạm hai vùng owner cần cả hai cùng review; `training.py` luôn cần hai reviewer vì sai ở đây làm lệch mọi nhánh.

## Kế hoạch 1 — Thành viên 1: Optimizer + Normalization + shared anchor

### Phạm vi sở hữu

- Code review chính: `model.py`, phần `build_optimizer()` trong `optimization.py`.
- Config: `member1_optimizer_norm.yaml`.
- Official workload: 14 unique configs × 2 seed = **28 runs**.
- Shared anchor do người 1 chạy, nhưng được dùng trong bốn nhánh.
- Phần báo cáo: “Optimizer” và “Normalization”.

### Danh sách run

- Optimizer: shared SGD+momentum anchor; SGD; Nesterov; RMSProp có momentum 0.9; Adam; AdamW = 6 cấu hình.
- Normalization bổ sung: BN batch 8/32; LN batch 8/32/128; GN batch 8/32/128 = 8 cấu hình. BN batch 128 lấy từ anchor.

### Lịch chi tiết

**Ngày 1**

1. Vẽ tay một BasicBlock: main path hai conv và shortcut.
2. In shape sau stem/stage1/2/3/4 và xác nhận 32→32→16→8→4.
3. Forward/backward cả BN/LN/GN trên batch giả; kiểm output `[N,10]`.
4. Đọc exact arguments của sáu optimizer, đặc biệt momentum/Nesterov/RMSProp và coupled-vs-decoupled WD.
5. Cùng người 2 xem LR finder; không tự chọn LR vì một curve đẹp.

**Ngày 2**

1. Nhận sáu LR pilot đã review; cập nhật matrix qua PR.
2. Dry-run queue; đối chiếu 28 commands và hai seed.
3. Chạy optimizer trước. Sau mỗi cặp seed, kiểm `status=completed`, split hash, commit, epoch count.
4. Vẽ nhanh loss; nếu diverge, đánh dấu pilot/protocol issue, không âm thầm đổi LR giữa official runs.

**Ngày 3**

1. Chạy normalization theo batch 8→32→128; theo dõi quota vì batch 8 chậm nhất.
2. Xác nhận số trainable parameters của BN/LN/GN bằng nhau; giải thích BN có thêm running-stat buffers không trainable.
3. Chỉ so runtime giữa cùng GPU; ghi OOM nếu batch 128 rồi xử lý protocol công khai.

**Sáng ngày 4**

1. Aggregate; xác nhận 12 optimizer runs + 18 normalization comparison runs khi tính anchor.
2. Bàn giao hai figure optimizer, một figure norm-vs-batch và bảng mean±std.
3. Viết kết quả theo mẫu “quan sát → mức dao động → cơ chế có thể giải thích → giới hạn”.
4. Chuẩn bị 3 case chưa đủ bằng chứng nếu std lớn.

### Definition of done

- Không thiếu seed/config, không khác split/protocol.
- Sáu optimizer có loss curve theo epoch/step và accuracy mean±std.
- 3×3 normalization grid đầy đủ.
- Giải thích được tại sao BN phụ thuộc batch, LN/GN không dùng batch statistics.
- Giải thích được vì sao AdamW cần WD dương để khác Adam trong khảo sát này.

### Câu hỏi bảo vệ tự luyện

1. Nesterov khác momentum ở điểm nhìn trước gradient thế nào?
2. RMSProp và Adam lưu moving average nào?
3. Vì sao cùng seed không đồng nghĩa giống bit tuyệt đối trên GPU khác?
4. LayerNorm2d trong repo normalize trên chiều nào?
5. Batch size thay đổi cả statistics lẫn số update/epoch; nhóm tách hai hiệu ứng ra sao?

## Kế hoạch 2 — Thành viên 2: LR schedule + LR finder + tổng hợp/plot

### Phạm vi sở hữu

- Code review chính: `LearningRateSchedule`, `lr_range_test.py`, `aggregate_results.py`, `plot_results.py`.
- Config: `member2_schedules.yaml`.
- Official workload: 5 unique configs × 2 seed = **10 runs**; cấu hình thứ sáu dùng shared anchor.
- Pilot bổ sung: LR range test công bằng cho sáu optimizer.
- Phần báo cáo: “Learning-rate schedule và warm-up”; hỗ trợ khung hình/bảng chung.

### Danh sách run

- Constant + warm-up.
- Step/no warm-up và step/warm-up.
- Cosine/no warm-up và cosine/warm-up.
- Constant/no warm-up tái sử dụng anchor.

### Lịch chi tiết

**Ngày 1**

1. Đọc công thức `factor(step)` và lập bảng LR cho step đầu/cuối warm-up, epoch decay và step cuối cosine.
2. Chạy LR range cho sáu optimizer với cùng start/end/steps, cùng seed, split, batch và model.
3. Không chọn đúng LR tại điểm loss thấp nhất; chọn vùng giảm ổn định trước divergence và ghi rule trước khi nhìn official validation.
4. Bàn giao sáu CSV/hình, LR đề xuất và lý do cho cả nhóm review.

**Ngày 2**

1. Sau protocol freeze, dry-run 10 commands.
2. Chạy lần lượt một seed cho đủ năm cấu hình để bắt lỗi sớm, sau đó seed còn lại.
3. Kiểm `metrics_step.csv` có LR đúng công thức và không reset sai ở ranh giới epoch.
4. Kiểm step schedule decay sau warm-up, cosine kết thúc gần `min_lr`.

**Ngày 3**

1. Hoàn tất retry có ghi lý do.
2. Chạy aggregator không strict để thấy thiếu seed, xử lý đến khi audit sạch.
3. Tạo loss-vs-step và accuracy mean±std; không làm mượt quá mạnh để che divergence (window 50 phải ghi trên nhãn).
4. Hỗ trợ người 1/3 tạo đủ tám hình từ cùng script.

**Sáng ngày 4**

1. Chạy `aggregate_results.py --strict`; lưu `master.csv`, `summary_mean_std.csv`, `audit.json`.
2. Đối chiếu ngẫu nhiên ba ô report với raw CSV.
3. Viết phần schedule: convergence, final/best accuracy, vai trò warm-up, giới hạn 20 epoch.
4. Xuất bảng config chính xác để phần phương pháp không lệch code.

### Definition of done

- LR finder có sáu curve và protocol chọn LR bằng validation/pilot, không test.
- Schedule grid đủ 3×2 khi cộng anchor.
- Loss-vs-step dùng global optimizer step, label rõ smoothing.
- Aggregator phát hiện thiếu seed/split mismatch/duplicate.
- Tất cả số trong bảng và hình có thể lần về run directory.

### Câu hỏi bảo vệ tự luyện

1. Warm-up giải quyết vấn đề gì ở các update đầu?
2. Scheduler được gọi trước hay sau optimizer step, và step 0 có LR bao nhiêu?
3. Step decay/cosine khác nhau về inductive bias thế nào?
4. Vì sao LR finder không trả ra một “LR tối ưu” tuyệt đối?
5. Sample std với n=2 được tính thế nào và vì sao chưa phải significance test?

## Kế hoạch 3 — Thành viên 3: Regularization + phân tích overfit + mở rộng

### Phạm vi sở hữu

- Code review chính: augmentation trong `data.py`, Dropout trong `model.py`, early stopping trong `training.py`.
- Config: `member3_regularization.yaml`.
- Official workload: 8 unique configs × 2 seed = **16 runs**; WD-only lấy từ shared anchor.
- Extension chỉ sau bắt buộc: label smoothing hoặc Mixup (ưu tiên label smoothing vì code đã có và dễ giải thích).
- Phần báo cáo: “Regularization và mở rộng”.

### Danh sách run

- None.
- Dropout 0.1/0.3/0.5.
- Augmentation.
- Early stopping.
- WD + augmentation.
- WD + Dropout 0.3 + augmentation + early stopping.
- WD-only tái sử dụng anchor.

### Lịch chi tiết

**Ngày 1**

1. Hiển thị vài ảnh trước/sau crop/flip/jitter và xác nhận validation không augment.
2. Xác nhận Dropout nằm sau GAP, chỉ hoạt động ở `model.train()` và tắt ở `model.eval()`.
3. Lập ví dụ early stopping bằng chuỗi val loss giả; xác nhận patience/min_delta.
4. Chốt “không dùng regularizer đang khảo sát” có WD=0, dropout=0, augmentation=false, early=false. BatchNorm vẫn tồn tại nên không gọi đây là mô hình hoàn toàn không có hiệu ứng regularization.

**Ngày 2**

1. Dry-run 16 commands; kiểm mỗi cấu hình chỉ đổi đúng treatment.
2. Chạy baseline không regularizer khảo sát, ba dropout và augmentation cho seed 42 trước; xem underfit/divergence.
3. Hoàn thành seed 2026; không đổi dropout location giữa runs.

**Ngày 3**

1. Chạy early stopping, WD+aug và combined cho hai seed.
2. Ghi epoch dừng và runtime; không so accuracy mà bỏ qua ngân sách thấp hơn.
3. Tạo accuracy chart, generalization-gap chart, individual-vs-combined chart.
4. Chỉ khi audit bắt buộc sạch, thêm label smoothing với protocol version `extension-v1`; không trộn vào bảng bắt buộc.

**Sáng ngày 4**

1. Phân loại từng phương pháp là underfit/overfit/không rõ dựa trên cả loss và gap.
2. So riêng lẻ với kết hợp; không kết luận đóng góp cộng tuyến từ một combined run.
3. Viết phần regularization và limitation: augmentation thay phân phối input; early stopping thay budget; WD semantics tùy optimizer.
4. Bàn giao config, 16 run logs, ba hình, bảng mean±std, đoạn extension nếu có.

### Definition of done

- Grid 9 cấu hình đầy đủ khi cộng WD anchor.
- Validation transform không có randomness.
- Dropout ba mức cùng vị trí; early stopping có epoch dừng trong log.
- Phân tích dùng train loss/accuracy, val loss/accuracy và gap; không chỉ best accuracy.
- Extension tách protocol/table, không làm thiếu phần bắt buộc.

### Câu hỏi bảo vệ tự luyện

1. Weight decay khác Dropout về cơ chế nào?
2. Vì sao augmentation chỉ áp vào train?
3. Dropout được tắt khi inference bằng cơ chế nào của PyTorch?
4. Early stopping có phá điều kiện “cùng 20 epoch” không? Nhóm định nghĩa ngân sách ra sao?
5. Vì sao combined tốt hơn không chứng minh từng thành phần cộng độc lập?

## Ngày 4–5 — tích hợp và bảo vệ chéo

1. Người 2 chạy audit; người 1 và 3 đối chiếu raw log độc lập.
2. Mỗi người viết phần mình nhưng một người khác review kết luận và biểu đồ.
3. Điền và commit `final_selection.yaml` bằng validation; test đúng cấu hình đó ở cả seed 42/2026 rồi báo mean±std.
4. Ghép report 8–12 trang, slide 8–12, phân công tổng 100%, AI declaration.
5. Chạy repo từ môi trường sạch theo README.
6. Mỗi người trình bày thử 5 phút phần không thuộc mình; sửa chỗ không giải thích được.

### Ghép báo cáo 8–12 trang

| Mục | Owner bản nháp | Reviewer |
|---|---|---|
| 1. Giới thiệu + đóng góp | Thành viên 1 | Thành viên 3 |
| 2. Cơ sở lý thuyết | mỗi người viết kỹ thuật nhánh mình | review vòng tròn 1→2→3→1 |
| 3. Dữ liệu/split/augmentation | Thành viên 3 | Thành viên 1 |
| 4. Model + quy trình train | Thành viên 1 | Thành viên 2 |
| 5. Thiết lập thí nghiệm + bảng/hình | Thành viên 2 | cả nhóm đối chiếu raw log |
| 6. Phân tích/thảo luận | mỗi người viết nhánh mình | cả nhóm thống nhất độ mạnh kết luận |
| 7. Kết luận/khuyến nghị | Thành viên 2 ghép | Thành viên 1 và 3 |
| Tài liệu tham khảo/phụ lục/phân công/AI | Thành viên 3 | Thành viên 2 |

Slide 8–12 trang dùng cùng số liệu/hình đã sinh từ script; không tạo một bảng số thứ hai bằng tay.
