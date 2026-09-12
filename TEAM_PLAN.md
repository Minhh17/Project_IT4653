# Phân công Nhiệm vụ & Kế hoạch Triển khai (TEAM_PLAN)

## 1. Bảng Phân công Trách nhiệm & Tỷ lệ Đóng góp

Bảng thống kê thông tin nhân sự, phạm vi công việc chuyên môn và tỷ lệ đóng góp dự kiến phục vụ công tác nộp bài và bảo vệ đồ án.

| STT | Thành viên | Họ tên & MSSV | Phạm vi Chuyên môn Chịu trách nhiệm | Tỷ lệ Đóng góp |
| :-: | :--- | :--- | :--- | :-: |
| 1 | Thành viên 1 | Nguyễn Hải Minh — 20252595M | Kiến trúc ResNet-18, Thuật toán Tối ưu (Optimizers), Chuẩn hóa (Normalization) & Cấu hình Anchor | 34% |
| 2 | Thành viên 2 | Nguyễn Thị Len — 20252103M | Lịch trình Lập kế hoạch Tốc độ học (Schedules), Tích hợp Pipeline, Tổng hợp Nhật ký & Trực quan hóa | 33% |
| 3 | Thành viên 3 | Nguyễn Lâm Nghĩa — 20252596M | Xử lý Dữ liệu, Tăng cường Dữ liệu (Augmentation), Kỹ thuật Điều hòa (Regularization) & Kiểm định Protocol Test | 33% |
| **--** | **Tổng cộng** | | | **100%** |

---

## 2. Nguyên tắc Phối hợp & Quản lý Phiên bản

* **Kiến trúc Mã nguồn Chuẩn (Base Notebook):** Thành viên 2 chịu trách nhiệm quản lý, đóng gói và gắn mã phiên bản (`NOTEBOOK_VERSION`: `v1.0`, `v2.0`,...) cho tệp mã nguồn chính.
* **Môi trường Thực thi Song song:** Các thành viên khai thác tệp mã nguồn chuẩn trên môi trường Kaggle Notebook theo phương thức `Copy & Edit`.
* **Cấu hình Phân nhánh:** Mỗi thành viên thực thi các thí nghiệm bằng cách thay đổi hằng số phân luồng `MEMBER`, `PART`, và danh sách `RUN_IDS` tương ứng với phạm vi được giao.
* **Đồng bộ Luồng Xử lý Lõi:** Mọi cập nhật liên quan đến logic tính toán hoặc cấu trúc dữ liệu cốt lõi phải được chuyển cho Thành viên 2 để nghiệm thu và phát hành phiên bản mã nguồn mới trước khi áp dụng diện rộng.

---

## 3. Chi tiết Phân công Chuyên môn

### 3.1 Thành viên 1 – Tối ưu hóa, Chuẩn hóa & Khởi tạo Anchor

* **Phạm vi Nghiên cứu:**
  * Phân tích lý thuyết kiến trúc `BasicBlock`, đường kết nối tắt (shortcut connection) và cơ chế điều chỉnh ResNet-18 cho dữ liệu.
  * Phân tích chuyên sâu 06 thuật toán tối ưu hóa: SGD, SGD Momentum, Nesterov, RMSprop, Adam, và AdamW.
  * Thực hiện khảo sát (pilot) hoặc thiết lập thông số Tốc độ học (Learning Rate) tối ưu cho các thuật toán trước khi vận hành thực nghiệm chính thức.
  * Phân tích ảnh hưởng của các phương pháp chuẩn hóa (BatchNorm, LayerNorm, GroupNorm) kết hợp với kích thước Batch Size {8, 32, 128}.
  * Thực thi mô hình cơ sở dùng chung (Shared Anchor).
* **Khối lượng Thực nghiệm:** 14 cấu hình độc lập 2 seeds = **28 lượt huấn luyện (runs)**.
* **Sản phẩm Bàn giao:**
  * Nhật ký thực nghiệm: `summary_member1_*.csv`, log chi tiết từng epoch/step.
  * Đồ thị phân tích: 02 biểu đồ so sánh Optimizers, 01 biểu đồ đánh giá Normalization.
  * Soạn thảo tài liệu kỹ thuật: Phần phân tích Thuật toán Tối ưu và Kỹ thuật Chuẩn hóa.

### 3.2 Thành viên 2 – Lịch trình Lập kế hoạch LR, Tích hợp & Trực quan hóa

* **Phạm vi Nghiên cứu:**
  * Thiết lập công thức toán học cho các lịch trình LR: Constant, Step Decay, Cosine Annealing kết hợp trạng thái Warm-up.
  * Theo dõi và kiểm định giá trị LR thực tế ghi nhận trong log theo từng Epoch.
  * Xây dựng module tự động nạp tập hợp CSV từ các thành viên, tính toán thống kê Mean ± Std và trích xuất tệp `mean_std.csv`.
  * Khởi tạo hệ thống 06 biểu đồ cốt lõi phục vụ báo cáo.
  * Quản lý phiên bản tệp mã nguồn chuẩn (`NOTEBOOK_VERSION`).
* **Khối lượng Thực nghiệm:** 05 cấu hình độc lập 2 seeds = **10 lượt huấn luyện (runs)** (cấu hình Constant/No Warm-up kế thừa từ Anchor).
* **Sản phẩm Bàn giao:**
  * Nhật ký thực nghiệm: `summary_member2_*.csv`, log chi tiết từng epoch/step.
  * Đồ thị & Bảng biểu: 02 biểu đồ phân tích LR Schedules và Bảng thống kê tổng hợp `mean_std.csv`.
  * Soạn thảo tài liệu kỹ thuật: Phần phân tích Lịch trình LR và Thiết lập Thực nghiệm.

### 3.3 Thành viên 3 – Dữ liệu & Kỹ thuật Điều hòa (Regularization)

* **Phạm vi Nghiên cứu:**
  * Phân tích phương pháp phân chia dữ liệu 45k/5k và nguyên tắc loại bỏ Data Augmentation trên tập Validation.
  * Phân tích cơ chế của các kỹ thuật Điều hòa: Random Crop/Flip/Jitter, Dropout, Weight Decay, và Early Stopping.
  * Đánh giá hiệu năng của từng kỹ thuật riêng lẻ và cấu hình kết hợp (Combined Regularization).
  * Giám sát tính toàn vẹn của quy trình đánh giá: đảm bảo dữ liệu Test không bị biến đổi (ngoại trừ Normalization), không bị tráo đổi (no shuffle), và không tham gia vào quá trình tinh chỉnh tham số hay dừng sớm.
* **Khối lượng Thực nghiệm:** 07 cấu hình độc lập 2 seeds = **14 lượt huấn luyện (runs)** (cấu hình Weight Decay đơn lẻ kế thừa từ Anchor).
* **Sản phẩm Bàn giao:**
  * Nhật ký thực nghiệm: `summary_member3_*.csv`, log chi tiết từng epoch/step.
  * Đồ thị phân tích: Hệ thống biểu đồ đánh giá các kỹ thuật Regularization.
  * Soạn thảo tài liệu kỹ thuật: Phần phân tích Tiền xử lý Dữ liệu và Regularization.

---

## 4. Kế hoạch Tiến độ (Implementation Timeline)

+-------------------------------------------------------------------------------+
| GIAI ĐOẠN 1: KHỞI TẠO VÀ KIỂM THỬ PIPELINE                            |
+-------------------------------------------------------------------------------+
| - Thiết lập môi trường Kaggle GPU, nạp dataset CIFAR-10 chuẩn.                |
| - Kiểm tra luồng tính toán với DEBUG=True trên 01 epoch sample.               |
| - Chốt danh mục tham số (LR, Baseline, Notebook Version v1.0).                |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
| GIAI ĐOẠN 2: THỰC THI THÍ NGHIỆM SONG SONG                   |
+-------------------------------------------------------------------------------+
| - Chuyển DEBUG=False, phân chia RUN_IDS theo phân đoạn (part1, part2...).     |
| - Thực thi 52 lượt huấn luyện chính thức trên 02 seeds cố định (42, 2026).     |
| - Trích xuất và lưu trữ dữ liệu thô (.csv) sau mỗi phiên chạy thành công.      |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
| GIAI ĐOẠN 3: TỔNG HỢP VÀ PHÂN TÍCH DỮ LIỆU                            |
+-------------------------------------------------------------------------------+
| - Tích hợp toàn bộ file log .csv vào Notebook xử lý dữ liệu chung.            |
| - Tự động tính toán chỉ số Mean ± Std và xuất 06 biểu đồ phân tích.           |
| - Tổng hợp kết quả, đánh giá tính ổn định thống kê giữa các seeds.            |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
| GIAI ĐOẠN 4: NGHIỆM THU VÀ HOÀN THIỆN HỒ SƠ                          |
+-------------------------------------------------------------------------------+
| - Rà soát tính khớp nối giữa tập Validation và Test trên toàn bộ 26 cấu hình. |
| - Biên soạn Báo cáo kỹ thuật (Report PDF) và Slide thuyết minh.               |
| - Đóng gói Mã nguồn, Nhật ký thực nghiệm (.csv) và đưa lên GitHub Repository. |
+-------------------------------------------------------------------------------+