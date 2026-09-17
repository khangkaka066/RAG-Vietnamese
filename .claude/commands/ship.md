---
description: Điều phối dây chuyền Planner → Coder → Tester → Codex Reviewer cho một task cụ thể.
---

Bạn là điều phối viên (orchestrator) cho dây chuyền agent. Task cần thực hiện:

$ARGUMENTS

Thực hiện tuần tự các bước sau, KHÔNG bỏ bước, KHÔNG tự ý chạy `git merge` hay
`git push` ở bất kỳ đâu:

## Bước 0 — Chuẩn bị nhánh
1. Chạy `git status --short`. Nếu có thay đổi chưa commit không liên quan tới
   task này, dừng lại và báo cho người dùng trước khi tiếp tục.
2. Tạo slug ngắn gọn không dấu từ nội dung task, tạo và checkout nhánh
   `feature/<slug>` bằng `git checkout -b feature/<slug>`.
3. Tạo thư mục `.bangiao/` nếu chưa có.

## Bước 1 — Planner
Gọi Agent tool với `subagent_type: "planner"`, truyền nguyên văn task ở trên.
Sau khi xong, đọc `.bangiao/ke-hoach.md`.
- Nếu file có dòng bắt đầu bằng `CÂU HỎI CÒN BỎ NGỎ:` → DỪNG NGAY, hỏi lại
  người dùng câu hỏi đó (dùng AskUserQuestion hoặc hỏi trực tiếp), không đi
  tiếp các bước sau cho tới khi có câu trả lời.

## Bước 2 — Coder
Gọi Agent tool với `subagent_type: "coder"`. Sau khi xong, đọc
`.bangiao/thay-doi.md` để nắm được các file đã đổi.

## Bước 3 — Tester
Gọi Agent tool với `subagent_type: "tester"`. Sau khi xong, đọc
`.bangiao/ket-qua-test.md`.
- Nếu có dòng `TEST ROT:` → quay lại **Bước 2** (gọi lại Coder, kèm nội dung
  lỗi để sửa). Đếm số vòng lặp — tối đa 3 vòng. Nếu vẫn fail sau 3 vòng, dừng
  lại và báo cáo chi tiết cho người dùng, không tự ý thử thêm.

## Bước 4 — Codex Review
Chạy `bash scripts/codex_review.sh`. Sau khi xong, đọc `.bangiao/danh-gia.md`
và tìm dòng `PHAN QUYET: ...`.

- **`PHAN QUYET: CHOT`** → đi tới Bước 5.
- **`PHAN QUYET: CAN SUA`** → quay lại **Bước 2** (gọi lại Coder, kèm toàn bộ
  nội dung `.bangiao/danh-gia.md` làm feedback). Đếm số vòng lặp — tối đa 3
  vòng riêng cho review. Nếu vẫn "CAN SUA" sau 3 vòng, dừng lại và báo cáo cho
  người dùng.
- **`PHAN QUYET: CHAN`** → DỪNG NGAY LẬP TỨC, báo cáo chi tiết lý do cho người
  dùng. KHÔNG tự động thử sửa tiếp.

## Bước 5 — Commit (không merge)
Sau khi review CHOT: chạy `git add -A` rồi `git commit` với message mô tả
task, kết thúc bằng dòng attribution chuẩn của Claude Code. KHÔNG chạy
`git merge` hay `git push` — báo cho người dùng biết nhánh `feature/<slug>`
đã sẵn sàng, họ tự xem diff và merge tay.

## Báo cáo cuối
Dù kết thúc ở bước nào, luôn tóm tắt ngắn gọn cho người dùng: đã đi tới bước
nào, kết quả ra sao, và bước tiếp theo người dùng cần làm (nếu có).
