#!/usr/bin/env bash
# Gọi Codex CLI làm Reviewer độc lập cho dây chuyền /ship.
# Codex KHÔNG được sửa code (chạy read-only), chỉ review và ghi verdict.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p .bangiao

# Intent-to-add CHỈ các file untracked (không đụng file đã staged sẵn) để
# `git diff HEAD` bên trong Codex cũng thấy được file mới, bù cho việc dùng
# `codex exec` thường (không phải subcommand `review`) nên phải tự yêu cầu
# model chạy git diff lấy thay đổi thay vì dựa vào --uncommitted.
UNTRACKED_FILES=()
while IFS= read -r -d '' f; do
  UNTRACKED_FILES+=("$f")
done < <(git status --porcelain -z | awk -v RS='\0' 'substr($0,1,2)=="??"{print substr($0,4)}' | tr '\n' '\0')
if [ "${#UNTRACKED_FILES[@]}" -gt 0 ]; then
  git add -N -- "${UNTRACKED_FILES[@]}"
fi

SCHEMA_FILE="$(mktemp -t codex_review_schema.XXXXXX.json)"
JSON_OUT="$(mktemp -t codex_review_out.XXXXXX.json)"
# Trap luôn chạy dù codex exec thành công hay lỗi (set -e), để không bao giờ
# để sót intent-to-add trong index; chỉ reset đúng các file untracked vừa
# thêm ở trên, không đụng tới bất kỳ file nào người dùng đã tự staged trước đó.
cleanup() {
  rm -f "$SCHEMA_FILE" "$JSON_OUT"
  if [ "${#UNTRACKED_FILES[@]}" -gt 0 ]; then
    git reset --quiet -- "${UNTRACKED_FILES[@]}"
  fi
}
trap cleanup EXIT

cat > "$SCHEMA_FILE" <<'JSON'
{
  "type": "object",
  "additionalProperties": false,
  "required": ["verdict", "nhan_xet"],
  "properties": {
    "verdict": {
      "type": "string",
      "enum": ["CHOT", "CAN_SUA", "CHAN"]
    },
    "nhan_xet": {
      "type": "string",
      "description": "Toàn bộ nội dung review bằng tiếng Việt: đúng/sai so với kế hoạch, rủi ro bảo mật/logic, code smell đáng chú ý."
    }
  }
}
JSON

PROMPT="Bạn là Reviewer độc lập, KHÔNG được sửa code. Trước tiên chạy \`git status --short\`
và \`git diff HEAD\` (hoặc \`git diff --cached\` nếu đã staged) để lấy toàn bộ thay đổi chưa
commit, rồi đối chiếu với .bangiao/ke-hoach.md (kế hoạch gốc) và .bangiao/ket-qua-test.md
(kết quả test). Chỉ ra lỗi đúng/sai so với kế hoạch, rủi ro bảo mật/logic, code smell đáng chú ý.
Trả lời BẮT BUỘC đúng theo JSON schema đã cung cấp:
- 'verdict' = CHOT nếu thay đổi đạt yêu cầu, có thể commit ngay.
- 'verdict' = CAN_SUA nếu cần Coder sửa lại trước khi commit.
- 'verdict' = CHAN nếu có vấn đề nghiêm trọng, phải dừng pipeline ngay lập tức.
- 'nhan_xet' chứa toàn bộ nhận xét chi tiết bằng tiếng Việt (lý do, vị trí file:dòng nếu có)."

# Dùng `codex exec` (không phải subcommand `review`) vì `review` có sẵn
# template output riêng (danh sách finding theo mức [P1]/[P2]...) và bỏ qua
# hướng dẫn định dạng tự do được thêm vào prompt — đây là lý do gốc khiến
# dòng "PHAN QUYET: ..." liên tục bị thiếu ở các vòng review trước, kể cả
# vòng review gpt-5.5 đã được commit trong Phase 0 (xem .bangiao/danh-gia.md
# tại commit 40ff9bc). `--output-schema` ép model trả JSON đúng cấu trúc,
# 'verdict' là enum nên không phụ thuộc vào việc model có "nhớ" viết đúng
# một dòng text tự do hay không.
codex exec \
  -c sandbox_mode="read-only" \
  -m gpt-5.6-luna \
  -c model_reasoning_effort="high" \
  --output-schema "$SCHEMA_FILE" \
  -o "$JSON_OUT" \
  "$PROMPT"

# Dọn lại index: bỏ intent-to-add vừa thêm ở trên, đưa index về đúng HEAD.
# Working tree không đổi — các file untracked lại trở về trạng thái untracked,
# tránh để lại side effect trong `git status` sau khi script này chạy xong.
git reset --quiet -- .

VERDICT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$JSON_OUT")"
NHAN_XET="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["nhan_xet"])' "$JSON_OUT")"

case "$VERDICT" in
  CHOT) VERDICT_LINE="PHAN QUYET: CHOT" ;;
  CAN_SUA) VERDICT_LINE="PHAN QUYET: CAN SUA" ;;
  CHAN) VERDICT_LINE="PHAN QUYET: CHAN" ;;
  *)
    echo "Codex trả về verdict không hợp lệ: '$VERDICT'" >&2
    exit 1
    ;;
esac

{
  printf '%s\n\n' "$NHAN_XET"
  printf '%s\n' "$VERDICT_LINE"
} > .bangiao/danh-gia.md

cat .bangiao/danh-gia.md
