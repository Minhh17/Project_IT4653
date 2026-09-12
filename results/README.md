# Quy chuẩn Lưu trữ & Quản lý Nhật ký Thực nghiệm (Results Specification)

## 1. Tổng quan & Phạm vi Lưu trữ

Thư mục này quản lý toàn bộ **dữ liệu thực nghiệm chính thức** phục vụ việc trích xuất số liệu cho Báo cáo kỹ thuật (`report.pdf`) và Slide báo cáo (`slides.pdf`). 

**Nguyên tắc đóng gói:**
* **Chỉ lưu trữ:** Các tập tin nhật ký dữ liệu thô dạng `.csv`, dữ liệu tổng hợp thống kê, và đồ thị định dạng `.png`.
* **Tuyệt đối không đưa vào Git (phải được loại bỏ qua `.gitignore`):**
  * Tập dữ liệu thô CIFAR-10.
  * Các tập tin trọng số mô hình (`.pt`, `.pth`, checkpoints).
  * Nhật ký chạy thử nghiệm/kiểm thử luồng (`pilot_*.csv`).

---

## 2. Cấu trúc Thư mục Chuẩn (Target Directory Tree)

Sau khi hoàn tất toàn bộ 52 lượt huấn luyện chính thức, cấu trúc thư mục được đồng bộ như sau:

```text
results/
├── README.md                           # Tài liệu hướng dẫn quy chuẩn lưu trữ này
├── mean_std.csv                        # Bảng tổng hợp thống kê (Mean ± Std)
├── raw/                                # Nhật ký thực nghiệm dữ liệu thô
│   ├── summary_member1_part*.csv       # Nhật ký tổng hợp phần việc Thành viên 1
│   ├── summary_member2_part*.csv       # Nhật ký tổng hợp phần việc Thành viên 2
│   ├── summary_member3_part*.csv       # Nhật ký tổng hợp phần việc Thành viên 3
│   ├── epoch_log_member*.csv           # Log chỉ số chi tiết theo từng Epoch
│   └── step_log_member*.csv            # Log chỉ số chi tiết theo từng Iteration/Step
└── figures/                            # Biểu đồ phân tích hiệu năng (Tối thiểu 6 đồ thị)
    ├── 01_optimizer_loss.png           # So sánh Loss các Optimizers
    ├── 02_optimizer_accuracy.png       # So sánh Accuracy các Optimizers
    ├── 03_schedule_loss.png            # So sánh Loss các LR Schedules
    ├── 04_schedule_accuracy.png        # So sánh Accuracy các LR Schedules
    ├── 05_normalization.png            # Đánh giá ảnh hưởng của Normalization & Batch Size
    └── 06_regularization.png           # Đánh giá các kỹ thuật Regularization
