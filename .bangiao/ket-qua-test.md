# Kết quả test: Phase 1 — Embedding retrieval (vòng sửa sau Codex review 2)

## Lệnh đã chạy
- `python3 -m pytest -v` (chạy trực tiếp qua `pytest.main`, không qua wrapper rtk, để lấy đầy đủ tên test)
- `python3 <script kiểm tra độc lập tự viết>` — tái hiện adversarial scenario của Codex, không dùng test có sẵn của Coder
- `python3 <script đo cosine thật>` — đo lại cosine similarity trên `bkai-foundation-models/vietnamese-bi-encoder` với `data/knowledge_base.jsonl` + `data/eval.jsonl`
- `python3 scripts/run_eval.py` (mặc định lexical)

## Kết quả

### 1. pytest toàn bộ
21 passed, 0 failed, 0 skipped (sentence-transformers có sẵn trong môi trường nên nhóm B (real model) cũng chạy, không skip). Khớp với con số Coder báo cáo (21 test PASS, có thêm 1 test adversarial mới so với vòng trước).

### 2. Fix HybridRetriever alpha=0.0/1.0 bypass — ĐÃ XÁC NHẬN ĐÚNG, độc lập
- Đọc `src/vsf_rag/retrieval.py`: xác nhận `alpha == 0.0` trả thẳng `self.lexical.search(query, top_k=top_k)` (dòng 154-155) và `alpha == 1.0` trả thẳng `self.embedding.search(query, top_k=top_k)` (dòng 156-157), không đi qua công thức fusion `min(lexical_score, 1.0)` nữa. Vùng `0 < alpha < 1` vẫn giữ cap này (có lý do: tránh 1 nhánh áp đảo nhánh kia trong vùng blend) — hợp lý.
- Tự viết script độc lập (không dùng lại test của Coder) tái hiện đúng bug Codex mô tả: 3 documents (`filler_*` + `zzz_lo`, `bbb_hi`, `aaa_mid`) với lexical score được đẩy lên hẳn 6.27 / 5.55 / 2.25 (đều > 1.0, thậm chí vượt xa mốc trước kia dùng ~1.0x để test biên). Dùng một `BoomEmbedding` giả sẽ raise `AssertionError` nếu `embedding.search` bị gọi (đảm bảo bypass thật, không phải trùng hợp trả cùng kết quả).
- Kết quả: `HybridRetriever(alpha=0.0).search(...)` trả về đúng thứ tự VÀ đúng điểm số y hệt `LexicalRetriever.search(...)` thuần (`['aaa_mid', 'bbb_hi', 'zzz_lo']`, scores `[6.27, 5.55, 2.25]`) — không bị clip về 1.0, không bị tie-break sai theo id. Test tương tự cho alpha=1.0 (embedding-only) cũng pass.
- Test adversarial `test_hybrid_alpha_zero_matches_lexical_only_ranking_above_unit_score` của Coder trong `tests/test_embedding_retrieval.py` (dòng 209-247) cũng đã đọc qua: kịch bản dựng đúng như Codex mô tả (2 tài liệu score > 1.0, tài liệu điểm cao hơn có id sắp xếp sau theo alphabet), không phải test hời hợt — assert cả sanity-check của chính kịch bản trước khi assert kết quả.
- **Kết luận: bug thật đã được Codex reproduce ở round 2 đã được fix đúng, xác nhận độc lập không chỉ dựa vào báo cáo của Coder.**

### 3. minimum_score = 0.3 vs 0.5 — ĐÃ XÁC NHẬN, số liệu không bịa
- `src/vsf_rag/retrieval_embedding.py` dòng 69-76: comment giải thích rõ lý do 0.3 thay vì 0.5, dẫn số đo out-of-domain ~0.2063 / in-domain ~0.3364, khớp với `.bangiao/thay-doi.md`.
- Tự đo lại độc lập bằng model thật `bkai-foundation-models/vietnamese-bi-encoder` (không dùng script/số liệu có sẵn của Coder):
  - Out-of-domain (4 câu hỏi tự chọn, hoàn toàn không liên quan tới knowledge base: nấu phở, giá vàng, chăm cây, luật bóng đá): cosine cao nhất quan sát được = **0.1983** (khớp cùng cấp độ với 0.2063 Coder báo cáo; chênh lệch nhỏ vì dùng câu hỏi OOD khác, không phải sai số).
  - In-domain (toàn bộ 20 case trong `data/eval.jsonl`, so với `expected_doc_ids`): cosine thấp nhất quan sát được = **0.3364** đúng chính xác tại case `eval-003` — **khớp tuyệt đối con số Coder báo cáo (0.3364)**, đây là bằng chứng mạnh số liệu không phải bịa.
  - Khoảng trống 0.1983–0.3364 (hoặc 0.2063–0.3364 với query gốc của Coder) đều nằm rõ ràng trên/dưới ngưỡng 0.3 → ngưỡng 0.3 hợp lý, không cắt nhầm in-domain, không lọt nhầm out-of-domain.

### 4. Fixture test đã thu hẹp except clause — ĐÃ XÁC NHẬN
`tests/test_embedding_retrieval.py`, hàm `_real_embedding_retriever_or_skip` (dòng 287+): không còn `except Exception: pytest.skip(...)` bao trùm. Đã đọc phần except cụ thể, chỉ bắt các lỗi đặc trưng cho môi trường thiếu mạng/model (HfHubHTTPError, LocalEntryNotFoundError, OSError, ConnectionError, TimeoutError) như báo cáo mô tả — lỗi tích hợp thật (ví dụ ValueError từ logic validate) sẽ làm test FAIL thay vì bị skip nhầm.

### 5. `python scripts/run_eval.py` (mặc định lexical) — KHÔNG REGRESSION
`retrieval_hit_rate: 1.0`, `citation_coverage: 1.0`, `answer_keyword_coverage: 1.0` trên toàn bộ 20 case, tất cả `status: "OK"`. Khớp baseline trước đó, không có regression.

## Chi tiết lỗi (nếu có)
Không có. Tất cả 5 mục kiểm tra đều pass khi xác nhận độc lập (không chỉ dựa vào báo cáo/test sẵn có của Coder). Không phát hiện test rỗng, test giả, hay số liệu bịa đặt. Fix cho bug alpha boundary của Codex là đúng và đã được tôi tái hiện lại bằng kịch bản độc lập với dữ liệu/setup khác hoàn toàn so với test của Coder, cho kết quả nhất quán.
