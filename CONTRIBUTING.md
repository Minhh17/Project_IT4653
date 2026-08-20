# Quy ước làm việc nhóm

## Branch và ownership

- `main`: chỉ code/protocol đã review.
- `member1/optimizer-normalization`
- `member2/schedule-analysis`
- `member3/regularization`
- Fix chung ngắn: `fix/<mo-ta>`.

Không để logic train trong notebook. Notebook chỉ setup và gọi CLI.

## Một pull request hợp lệ

1. Nêu yếu tố khoa học nào thay đổi và vì sao.
2. Liệt kê file/config bị ảnh hưởng.
3. Chạy `python scripts/check_configs.py` và unit tests.
4. Nếu đổi pipeline, đính kèm fake smoke output.
5. Nếu đổi config sau freeze, tăng protocol version và không trộn run cũ.
6. Có ít nhất một reviewer; config khoa học có hai reviewer.

## Commit gợi ý

- `feat(model): add explicit LayerNorm2d`
- `exp(schedule): freeze warmup at 2 epochs`
- `fix(data): keep validation transform deterministic`
- `docs(report): explain noisy two-seed comparison`

Không commit `data/`, checkpoint hoặc run thất bại quá lớn. `runs/` và `results/` được ignore trong lúc chạy để `git_status` phản ánh đúng thay đổi source/config. Sau khi audit sạch, force-add **đúng các** CSV/JSON và figure dùng trong báo cáo (không force-add checkpoint), ví dụ `git add -f results/master.csv results/summary_mean_std.csv results/audit.json results/figures/*.png`. Mỗi artifact phải lần được về config/commit.

## Checklist review

- Chỉ một treatment đổi trong comparison này?
- Test set có bị đọc không?
- Seed/split/checkpoint rule còn giữ nguyên?
- Log schema và aggregator còn đọc được?
- Dòng mới có giải thích được khi bảo vệ?
- README/lệnh reproduce có cần cập nhật?
