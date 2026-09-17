Phần dữ liệu Phase 0 đạt yêu cầu: 29 documents, 20 eval cases; schema, id và required_terms hợp lệ. Kiểm tra độc lập cho thấy pytest 4 passed và run_eval có các metric đều 1.0.

Tuy nhiên chưa thể commit:
- Kế hoạch chỉ cho phép sửa data, test và tasks/todo.md (`.bangiao/ke-hoach.md:47-53`), nhưng diff còn sửa `scripts/codex_review.sh`, `.gitignore` và `.bangiao/danh-gia.md`.
- `scripts/codex_review.sh:57` đổi model đã pin từ `gpt-5.5` sang `gpt-5.6-luna`; kết quả test không kiểm tra wrapper. Model không tương thích có thể làm `/ship` thất bại trước khi tạo verdict.
- `git add -A -N` tại dòng 13 làm thay đổi index. Nếu Codex lỗi/bị ngắt, trap dòng 17 không reset index; nếu thành công, `git reset --quiet -- .` dòng 66 reset cả các thay đổi staged có sẵn, trái với cam kết review read-only.
- `.bangiao/danh-gia.md:5` ghi sai vị trí lỗi (`:25` thay vì dòng thực tế `:57`), cho thấy artifact review đã lỗi thời.

Không phát hiện lỗi logic retrieval/answering hoặc vấn đề bảo mật nghiêm trọng mức CHAN, nhưng cần Coder xử lý scope và wrapper trước khi commit.

PHAN QUYET: CAN SUA
