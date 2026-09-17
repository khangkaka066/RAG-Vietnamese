# Kết quả test: Mở rộng knowledge_base.jsonl và eval.jsonl (vòng sửa 2 sau review)

## Lệnh đã chạy
`python3 -m pytest tests/ -v`

Kèm các script kiểm tra thủ công (PYTHONPATH=src) để xác nhận từng tiêu chí review:
- Truy vấn "RAG hoạt động như thế nào?" qua `GroundedAnswerEngine`.
- Case eval index 13 (dòng ~14 trong `data/eval.jsonl`, `expected_doc_ids=["eval-003"]`, `required_terms=["faithfulness","answer relevancy"]`).
- Toàn bộ `evaluate()` report trên `data/eval.jsonl`.

## Kết quả
- `pytest tests/`: 4 passed, 0 failed.
- Query "RAG hoạt động như thế nào?" → top-1 retrieved document id = `rag-001` (đúng yêu cầu P2 #1, không còn trả về `agent-001`).
- Case eval faithfulness (idx 13, `eval-003`) → top-1 retrieved = `eval-003` (đúng expected_doc_ids); answer sinh ra chứa đầy đủ cả hai required_terms: "faithfulness" và "answer relevancy" (đúng yêu cầu P2 #2).
- `evaluate()` report tổng: `cases=20` (bằng `len(eval_cases)`), `retrieval_hit_rate=1.0`, `citation_coverage=1.0`, `answer_keyword_coverage=1.0`.
- `tests/test_retrieval.py` dòng 37: `assert report["cases"] == len(eval_cases)` — không còn hard-code `== 20`, dùng so sánh động đúng như yêu cầu P2 #3.
- Không phát hiện file test nào khác trong repo ngoài `tests/test_retrieval.py`; không có test nào bị phá vỡ.

Tất cả 3 vấn đề review (rag-001 ranking, eval-003 ranking, hard-coded test case count) đã được xác nhận sửa đúng và test pass toàn bộ.

## Chi tiết lỗi (nếu có)
Không có.
