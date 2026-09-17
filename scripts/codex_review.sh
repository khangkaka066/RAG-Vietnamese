#!/usr/bin/env bash
# Gọi Codex CLI làm Reviewer độc lập cho dây chuyền /ship.
# Codex KHÔNG được sửa code (chạy read-only), chỉ review và ghi verdict.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p .bangiao

# Intent-to-add untracked files (không stage nội dung) để `git diff HEAD`
# bên trong Codex cũng thấy được file mới, bù cho việc không dùng được
# --uncommitted cùng lúc với PROMPT tùy chỉnh (hai flag này loại trừ nhau).
git add -A -N

PROMPT="Bạn là Reviewer độc lập, KHÔNG được sửa code. Trước tiên chạy \`git status --short\`
và \`git diff HEAD\` (hoặc \`git diff --cached\` nếu đã staged) để lấy toàn bộ thay đổi chưa
commit, rồi đối chiếu với .bangiao/ke-hoach.md (kế hoạch gốc) và .bangiao/ket-qua-test.md
(kết quả test). Chỉ ra lỗi đúng/sai so với kế hoạch, rủi ro bảo mật/logic, code smell đáng chú ý.
Kết thúc câu trả lời BẮT BUỘC bằng đúng một trong ba dòng sau (không thêm ký tự nào khác trên dòng đó):
PHAN QUYET: CHOT
PHAN QUYET: CAN SUA
PHAN QUYET: CHAN"

codex exec review \
  -c sandbox_mode="read-only" \
  -m gpt-5.5 \
  -c model_reasoning_effort="high" \
  -o .bangiao/danh-gia.md \
  "$PROMPT"

# Dọn lại index: bỏ intent-to-add vừa thêm ở trên, đưa index về đúng HEAD.
# Working tree không đổi — các file untracked lại trở về trạng thái untracked,
# tránh để lại side effect trong `git status` sau khi script này chạy xong.
git reset --quiet -- .

cat .bangiao/danh-gia.md
