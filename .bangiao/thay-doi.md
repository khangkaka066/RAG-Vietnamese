# Thay đổi: Mở rộng knowledge_base.jsonl và eval.jsonl (Phase 0) — vòng sửa theo review

## File đã đổi
- `data/knowledge_base.jsonl` — Sửa nội dung doc `agent-001` (dòng 4) để khắc phục vấn đề [P2 #1]:
  đổi cụm "tác động bên ngoài" → "ảnh hưởng bên ngoài" và "tự động gọi" → "tự ý gọi", loại bỏ hoàn
  toàn token "động" khỏi doc này. Trước đó, `agent-001` là doc duy nhất khác `rag-001` chứa token
  "động" (từ "tác động"/"tự động"), nên với câu hỏi "RAG hoạt động như thế nào?" (chứa token hiếm
  "động"), TF-IDF xếp `agent-001` lên top-1 thay vì `rag-001`. Sau khi sửa, `rag-001` là top-1 duy
  nhất khớp câu hỏi này (đã kiểm chứng lại bằng script gọi trực tiếp `GroundedAnswerEngine`), giữ
  đúng hành vi case eval gốc `RAG hoạt động như thế nào?` → `rag-001`, `keyword_coverage` = 1.0.
  Không đổi phần còn lại của file.
- `data/eval.jsonl` — Sửa case về Faithfulness/answer relevancy (dòng 14) để khắc phục vấn đề
  [P2 #2]: đổi query từ "Faithfulness và answer relevancy dùng để đánh giá gì trong LLM?" (dùng
  nhiều từ chung với case `eval-001` như "đánh giá", "LLM") sang "Câu trả lời có bịa thêm thông tin
  ngoài ngữ cảnh và có giải quyết đúng câu hỏi được đặt ra hay không?" — dùng các cụm từ đặc trưng
  chỉ xuất hiện trong nội dung `eval-003` ("bịa", "giải quyết", "câu hỏi được đặt ra"). Đã kiểm
  chứng lại: engine trả về top-1 = `eval-003` (không còn bị `eval-001` chiếm top-1), câu trả lời
  sinh ra từ đúng text của `eval-003` nên chứa đủ cả hai `required_terms` ("faithfulness" và
  "answer relevancy"), `keyword_coverage` = 1.0. Giữ nguyên `expected_doc_ids` và `required_terms`.
- `tests/test_retrieval.py` (dòng ~34-38) — Khắc phục vấn đề [P2 #3]: bỏ hard-code
  `assert report["cases"] == 20`, thay bằng so sánh động
  `assert report["cases"] == len(eval_cases)` (dùng chung biến `eval_cases` đã load từ
  `load_eval_cases(...)` trước khi gọi `evaluate`), để test không vỡ khi số case trong
  `data/eval.jsonl` thay đổi hợp lệ.

## Ghi chú
- Đã tự kiểm tra lại bằng script gọi trực tiếp `LexicalRetriever`/`GroundedAnswerEngine` cho cả hai
  câu hỏi liên quan: "RAG hoạt động như thế nào?" → top-1 `rag-001`; câu hỏi faithfulness mới →
  top-1 `eval-003`, cả hai đều status `OK` và chứa đúng citation mong đợi.
- Đã chạy `python3 -m pytest tests/ -q` — toàn bộ 4 test pass (bao gồm
  `test_evaluation_report_has_quality_metrics` với số case động, hiện là 20).
- Không thay đổi gì khác ngoài 3 điểm review đã nêu; các phần khác của `data/knowledge_base.jsonl`
  và `data/eval.jsonl` (tổng số doc/case, các case còn lại) giữ nguyên như vòng trước.
