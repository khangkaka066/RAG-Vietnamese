# Kế hoạch: Mở rộng knowledge_base.jsonl và eval.jsonl (Phase 0)

## Mục tiêu
Nâng `data/knowledge_base.jsonl` từ 5 → ít nhất 25-30 documents đa dạng chủ đề, và
`data/eval.jsonl` từ 5 → 18-20 case hỏi tiếng Việt, sao cho số liệu eval
(retrieval hit-rate, citation coverage, keyword coverage) có ý nghĩa thống kê thay vì
chỉ 5 mẫu. KHÔNG chọn LLM provider / embedding model ở task này. KHÔNG sửa logic
retrieval/answering — chỉ sửa dữ liệu + cập nhật assertion test bị hard-code theo số 5.

## Ràng buộc bắt buộc (đọc từ code, vi phạm là vỡ pipeline)

1. **Schema document cố định tuyệt đối.** `src/vsf_rag/retrieval.py:40` dùng
   `Document(**record)` với dataclass chỉ có 4 field: `id`, `title`, `source`, `text`.
   Thêm bất kỳ key nào (vd `tags`, `topic`) → `TypeError` → `ValueError("Invalid document at line N")`.
   → Mỗi dòng phải đúng 4 key này, không hơn không kém.

2. **Schema eval case:** `query` (bắt buộc, `evaluation.py:26` truy cập trực tiếp),
   `expected_doc_ids` (list), `required_terms` (list). Giữ nguyên 3 key này.

3. **`required_terms` phải là substring có thật trong câu trả lời.**
   `answering.py:59` sinh answer = `f"Theo tài liệu '{title}': {text}"`, và
   `evaluation.py:33` chấm bằng `term in answer_lower`. Vậy mọi term trong
   `required_terms` PHẢI xuất hiện (dạng lowercase, khớp chính xác ký tự kể cả dấu)
   trong `title` hoặc `text` của document trong `expected_doc_ids`. Nếu không,
   `answer_keyword_coverage` tụt vô cớ.

4. **Độ dài document phải tương đương bản hiện có (~35-60 token).**
   `retrieval.py:82` chia điểm cho `sqrt(len(tokens))`. Doc dài → điểm thấp →
   rớt dưới `minimum_score = 0.15` (`answering.py:28`) → trả về `UNAVAILABLE` →
   `citation_coverage` < 1.0. Viết doc 2-3 câu, đúng phong cách 5 doc hiện tại.

5. **Mỗi document phải có "từ khóa định danh" riêng.** Retriever là lexical TF-IDF;
   IDF giảm khi nhiều doc cùng chứa một từ. Nếu 10 doc đều nhồi "RAG", "đánh giá",
   "model" thì query mơ hồ sẽ bắt sai doc. Mỗi doc cần 2-3 thuật ngữ đặc trưng
   (vd `reranking`, `p95`, `drift`, `chunk overlap`, `rate limit`) và query eval
   tương ứng phải chứa chính các thuật ngữ đó.

6. **Không được làm hỏng 2 test regression hiện có:**
   - `tests/test_retrieval.py:24` — query `"RAG cần trích dẫn nguồn như thế nào?"`
     vẫn phải cho `rag-001` ở vị trí top-1. → Các doc mới về RAG KHÔNG được lặp lại
     cụm "trích dẫn" + "RAG" dày đặc; ưu tiên dùng từ khác (`nguồn tham chiếu`,
     `citation id`) để không cướp top-1 của `rag-001`.
   - `tests/test_retrieval.py:29-31` — query `"Thời tiết hôm nay ở sao Hỏa thế nào?"`
     vẫn phải ra `UNAVAILABLE`. → Tuyệt đối không dùng các từ `thời tiết`, `hôm nay`,
     `sao Hỏa` trong bất kỳ doc mới nào.

## File sẽ đụng tới
- `data/knowledge_base.jsonl` — thêm 20-25 dòng document mới (giữ nguyên 5 dòng cũ, append).
- `data/eval.jsonl` — thêm 13-15 case mới (giữ nguyên 5 case cũ, append).
- `tests/test_retrieval.py` — dòng 36 `assert report["cases"] == 5` sẽ sai sau khi mở rộng;
  đổi thành so với `len(load_eval_cases(...))` hoặc `>= 18`. Đây là file duy nhất
  ngoài `data/` được phép sửa.
- `tasks/todo.md` — tick `[x]` dòng 12 sau khi xong.

## Các bước cụ thể

1. Append 22 document mới vào `data/knowledge_base.jsonl`, giữ convention id
   `<prefix>-<số thứ tự 3 chữ số>` như bản cũ. Phân bổ chủ đề đề xuất (mỗi doc 1 chủ đề,
   không trùng thuật ngữ định danh):
   - `rag-002` chunking & overlap · `rag-003` hybrid search (BM25 + vector)
   - `rag-004` reranking cross-encoder · `rag-005` query rewriting / HyDE
   - `emb-001` embedding tiếng Việt & bi-encoder · `emb-002` chuẩn hóa dấu, tách từ tiếng Việt
   - `emb-003` cosine similarity & normalize vector
   - `vec-001` vector database & index (HNSW/IVF) · `vec-002` metadata filtering
   - `eval-002` hit-rate & MRR@k · `eval-003` faithfulness / answer relevancy
   - `eval-004` golden dataset & version hóa bộ test
   - `llm-001` prompt template & system prompt · `llm-002` hallucination & grounding
   - `llm-003` fine-tuning vs RAG (khi nào chọn cái nào) · `llm-004` context window & token budget
   - `mlops-002` experiment tracking (khái niệm run/param/metric, KHÔNG nêu tên công cụ cụ thể để
     không lấn Phase 0 đã chốt) · `mlops-003` data drift monitoring · `mlops-004` CI/CD cho ML
   - `agent-002` router quyết định RAG vs tool · `agent-003` tool schema & validate tham số
   - `api-002` rate limiting & caching · `api-003` observability, correlation id, tracing
   - `sec-001` PII redaction & prompt injection
   Điều chỉnh/ bỏ bớt tùy ý miễn đạt tối thiểu 25 doc tổng và không vi phạm ràng buộc 5, 6.

2. Với MỖI doc mới, viết `text` tiếng Việt 2-3 câu (~40-60 token), `title` ngắn tiếng Việt,
   `source` theo pattern `internal-demo/<slug-tiếng-anh>.md` (khớp 5 dòng cũ).

3. Append 14 eval case mới vào `data/eval.jsonl` (tổng 19). Quy tắc soạn mỗi case:
   - `query`: câu hỏi tiếng Việt tự nhiên, CHỨA ít nhất 2 thuật ngữ định danh của doc đích.
   - `expected_doc_ids`: thường 1 id; cho phép 2 id ở 2-3 case "đa nguồn"
     (`evaluation.py:29` chỉ cần giao khác rỗng).
   - `required_terms`: 2-3 cụm, mỗi cụm PHẢI copy nguyên văn (lowercase) từ title/text
     của doc đích — xem ràng buộc 3.
   - Phủ đủ các nhóm chủ đề ở bước 1, không dồn hết vào RAG.

4. Tự verify bằng mắt trước khi bàn giao: với từng case mới, mở doc đích và confirm
   từng `required_terms` xuất hiện literal trong `title + text`.

5. Sửa `tests/test_retrieval.py:36` thành assertion không hard-code số 5
   (vd `assert report["cases"] == len(load_eval_cases(ROOT / "data" / "eval.jsonl"))`
   hoặc `assert report["cases"] >= 18`). Giữ nguyên
   `retrieval_hit_rate >= 0.8` và `citation_coverage == 1.0` — đây chính là tín hiệu
   cho biết dữ liệu mới có chất lượng hay không; nếu fail thì sửa DỮ LIỆU, không hạ ngưỡng.

6. Chạy `pytest` và `python scripts/run_eval.py`, đọc `details` trong report để tìm case
   `retrieval_hit: false` hoặc `keyword_coverage < 1.0` rồi chỉnh query/term cho tới khi đạt.

7. Tick `[x]` dòng 12 trong `tasks/todo.md`.

## Tiêu chí done
- [ ] `data/knowledge_base.jsonl` có >= 25 dòng, mỗi dòng JSON hợp lệ đúng 4 key `id/title/source/text`, id duy nhất.
- [ ] `data/eval.jsonl` có >= 18 dòng, mỗi dòng đúng 3 key `query/expected_doc_ids/required_terms`.
- [ ] Mọi `expected_doc_ids` đều tồn tại trong knowledge base (không có id ma).
- [ ] Mọi `required_terms` xuất hiện literal trong title/text của doc đích.
- [ ] `pytest` pass toàn bộ, gồm cả 2 test regression `rag-001` top-1 và `UNAVAILABLE` sao Hỏa.
- [ ] `python scripts/run_eval.py` cho `retrieval_hit_rate >= 0.8` và `citation_coverage == 1.0` mà KHÔNG hạ ngưỡng test.
- [ ] Không sửa file nào trong `src/` (logic retrieval/answering giữ nguyên).
- [ ] `tasks/todo.md` dòng 12 đã tick.
