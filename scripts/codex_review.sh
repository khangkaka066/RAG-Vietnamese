#!/usr/bin/env bash
# Gọi Codex CLI làm Reviewer độc lập cho dây chuyền /ship.
# Codex KHÔNG được sửa code (chạy read-only), chỉ review và ghi verdict.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p .bangiao

PROMPT="Bạn là Reviewer độc lập, KHÔNG được sửa code. Đọc diff hiện tại (uncommitted changes),
đối chiếu với .bangiao/ke-hoach.md (kế hoạch gốc) và .bangiao/ket-qua-test.md (kết quả test).
Chỉ ra lỗi đúng/sai so với kế hoạch, rủi ro bảo mật/logic, code smell đáng chú ý.
Kết thúc câu trả lời BẮT BUỘC bằng đúng một trong ba dòng sau (không thêm ký tự nào khác trên dòng đó):
PHAN QUYET: CHOT
PHAN QUYET: CAN SUA
PHAN QUYET: CHAN"

codex exec review --uncommitted \
  -c sandbox_mode="read-only" \
  -o .bangiao/danh-gia.md \
  "$PROMPT"

cat .bangiao/danh-gia.md
