# Thay đổi: Fix HybridRetriever alpha boundary contract + minimum_score docs (vòng sửa sau Codex review 2/3)

## File đã đổi
- `src/vsf_rag/retrieval.py` — Fix lỗi logic tại `HybridRetriever.search` mà Codex tìm thấy: `min(lexical_score, 1.0)` phá contract "alpha=0.0 phải tương đương lexical-only" (2 tài liệu điểm 1.97/1.85 đều bị cắt về 1.0 rồi tie-break sai theo id). Sửa bằng cách bypass hoàn toàn công thức fusion tại 2 đầu mút: khi `alpha == 0.0`, `search()` trả thẳng `self.lexical.search(query, top_k=top_k)`; khi `alpha == 1.0`, trả thẳng `self.embedding.search(query, top_k=top_k)`. Với `0 < alpha < 1`, giữ nguyên công thức trộn hiện tại (`alpha * cosine + (1 - alpha) * min(lexical_score, 1.0)`) — cap này chỉ còn ảnh hưởng vùng blend giữa 2 nhánh (nơi nó có lý do tồn tại: tránh 1 nhánh áp đảo nhánh kia), không còn ảnh hưởng tới 2 đầu mút vốn là nơi Codex phát hiện lỗi. Cập nhật docstring của `HybridRetriever` giải thích rõ vì sao bypass ở biên là cách đảm bảo đúng 100% thay vì cố chuẩn hoá công thức.
- `src/vsf_rag/retrieval_embedding.py` — Thêm comment tại nơi khai báo `minimum_score: float = 0.3` giải thích rõ lý do lệch so với đề xuất ban đầu 0.5 trong kế hoạch (xem số đo bên dưới).
- `tests/test_embedding_retrieval.py`:
  - Thêm test mới `test_hybrid_alpha_zero_matches_lexical_only_ranking_above_unit_score` — test adversarial tái hiện đúng kịch bản Codex mô tả: 3 tài liệu, trong đó 2 tài liệu ("mid" ~1.07, "hi" ~1.06) có điểm lexical > 1.0, với "mid" (điểm cao hơn) có id sắp xếp *sau* "hi" theo alphabet — đúng kịch bản khiến `min(score, 1.0)` + tie-break theo id trả sai thứ tự. Test verify `HybridRetriever(alpha=0.0).search(...)` trả về đúng y hệt thứ tự `lexical.search(...)` thuần (`["mid", "hi", "lo"]`), không bị cắt/tie-break sai.
  - Thu hẹp `except Exception: pytest.skip(...)` trong `_real_embedding_retriever_or_skip` (dòng ~246-252 cũ) thành bắt cụ thể `HfHubHTTPError`, `LocalEntryNotFoundError` (từ `huggingface_hub.utils`), `OSError`, `ConnectionError`, `TimeoutError` — các lỗi này đặc trưng cho môi trường không tải được model (không mạng, không cache local). Lỗi tích hợp thật (ví dụ `ValueError` từ logic validate của `EmbeddingRetriever`) giờ sẽ làm test FAIL thay vì bị skip nhầm thành "thiếu môi trường".

## Số đo cosine distribution (bằng chứng cho minimum_score=0.3, đo bằng model thật `bkai-foundation-models/vietnamese-bi-encoder` trên `data/knowledge_base.jsonl` + `data/eval.jsonl`)
- Out-of-domain (câu hỏi không liên quan tới knowledge base): cosine similarity cao nhất quan sát được ≈ **0.2063**.
- In-domain (câu hỏi thuộc eval set, có tài liệu liên quan thật trong knowledge base): cosine similarity thấp nhất quan sát được ≈ **0.3364**.
- Khoảng trống phân tách rõ ràng nằm giữa 0.2063 và 0.3364 → chọn ngưỡng **0.3** (nằm giữa khoảng trống, không cắt nhầm in-domain, không lọt nhầm out-of-domain).
- Nếu dùng 0.5 (giá trị đề xuất ban đầu trong kế hoạch trước khi đo thực tế) sẽ loại nhầm case `eval-003` hợp lệ (cosine của nó < 0.5 nhưng > 0.3364, tức vẫn nằm trong vùng in-domain thật).
- Kết luận: **giữ nguyên 0.3` trong code là đúng**, không đổi về 0.5 theo kế hoạch gốc; đã bổ sung comment tại chỗ khai báo để lý do này không bị mất khi đọc code độc lập với `.bangiao/`.

## Kết quả kiểm thử
- `python3 -m pytest -q`: 21 test PASS (thêm 1 test adversarial mới so với vòng trước, không xoá test nào).
- `python3 scripts/run_eval.py` (mặc định lexical): 20 case, hit-rate/citation/keyword đều 1.0 — không có regression.

## Ghi chú
- Không đụng tới file/logic ngoài phạm vi 3 việc yêu cầu trong vòng review này (không sửa `api.py`, `answering.py`, `pyproject.toml`, README, v.v.).
- Đánh đổi: với `0 < alpha < 1`, cap `min(lexical_score, 1.0)` trong vùng blend vẫn giữ nguyên như thiết kế trước (không đổi công thức trộn giữa) — theo đúng khuyến nghị của review là ưu tiên bypass đơn giản ở 2 đầu mút thay vì thiết kế lại toàn bộ công thức chuẩn hoá cho vùng giữa. Nếu sau này cần alpha trung gian với contract chặt hơn nữa, có thể cân nhắc chuẩn hoá riêng biệt (ví dụ min-max theo từng nhánh trước khi trộn) — không nằm trong phạm vi vòng sửa này.
- TODO còn lại (ngoài phạm vi vòng này, chỉ ghi chú cho người đọc sau): chưa có bộ đo cosine distribution tự động hoá dưới dạng script/test riêng — số đo ở trên được đo thủ công một lần trong quá trình review trước; nếu model hoặc dữ liệu eval thay đổi trong tương lai, nên đo lại trước khi tin tưởng ngưỡng 0.3.
