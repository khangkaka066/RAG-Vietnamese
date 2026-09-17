# Kế hoạch: Phase 1 — Embedding retrieval + Hybrid (thay lexical-only)

## Mục tiêu
Bổ sung `EmbeddingRetriever` (sentence-transformers, model
`bkai-foundation-models/vietnamese-bi-encoder`, output cosine similarity) bên cạnh
`LexicalRetriever` hiện có, thêm `HybridRetriever` kết hợp điểm hai bên, cho phép chọn
retriever qua config/factory, và thêm unit test so sánh hit-rate lexical vs embedding vs
hybrid trên `data/eval.jsonl`.

**Ràng buộc xuyên suốt**: mặc định của app/CI/Docker vẫn là `lexical` — không được làm
CI nặng thêm (torch ~800MB) hay phá 4 test regression hiện có. `sentence-transformers`
là **optional dependency**, import **lazy**, test embedding **skip** khi chưa cài / không
tải được model.

---

## Đọc trước: các ràng buộc rút ra từ code hiện tại

1. **Contract của `SearchResult`** (`src/vsf_rag/retrieval.py:26-30`): `document`,
   `score`, `matched_terms`. `answering.py:35-44` và `evaluation.py:27` đọc trực tiếp
   `result.score`, `result.document.id/title/source`, `result.matched_terms`.
   → Retriever mới **phải** trả về đúng kiểu `SearchResult` này, không tạo kiểu riêng.

2. **Ngưỡng `minimum_score = 0.15` đang hard-code cho thang điểm lexical**
   (`answering.py:28`). Lexical score = tf-idf / sqrt(len(tokens)), thực tế ~0.1-1.5.
   **Cosine similarity nằm trong [-1, 1]**, câu không liên quan tiếng Việt vẫn thường
   cho 0.2-0.4. Nếu dùng chung ngưỡng 0.15 cho embedding thì test
   `tests/test_retrieval.py:28-31` ("Thời tiết hôm nay ở sao Hỏa") sẽ **không còn ra
   `UNAVAILABLE`** → đây là cái bẫy chính của Phase này.
   → Ngưỡng phải thuộc về retriever, không thuộc về engine.

3. **`answering.py:28` type-hint `retriever: LexicalRetriever`** — cần đổi sang một
   Protocol chung, nếu không mypy/đọc code sẽ hiểu sai (runtime vẫn chạy được).

4. **Điểm hybrid không được min-max normalize theo từng query.** Nếu normalize, doc tốt
   nhất luôn = 1.0 kể cả khi query hoàn toàn không liên quan → `UNAVAILABLE` chết hẳn.
   Điểm hybrid phải giữ **thang tuyệt đối**.

5. **CI (`.github/workflows/ci.yml:16-18`) chạy `pip install -e '.[dev]'` rồi
   `pytest -q` + `python scripts/run_eval.py`** — cả hai phải pass khi KHÔNG có
   sentence-transformers.

---

## Quyết định thiết kế (đã chốt, coder không cần hỏi lại)

- **File mới `src/vsf_rag/retrieval_embedding.py`** chứa `EmbeddingRetriever` để mọi
  import nặng bị cô lập; `retrieval.py` giữ nguyên "zero dependency".
- **Dependency injection cho encoder**: `EmbeddingRetriever(documents, encoder=None)`.
  `encoder` là callable `(list[str]) -> list[Sequence[float]]`. Khi `None` thì lazy tạo
  `SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")` bên trong
  `__init__` (import bên trong hàm, không ở top-level module).
  → Nhờ vậy test cosine/top-k chạy được offline với fake encoder, không cần torch.
- **Cosine tính bằng `math` thuần** trên `list[float]` (chuẩn hóa vector về độ dài 1 ngay
  khi encode, rồi dot product). KHÔNG thêm numpy làm hard dependency.
- **Ngưỡng theo retriever**: mỗi retriever có attribute `minimum_score`.
  `LexicalRetriever.minimum_score = 0.15` (giữ nguyên hành vi cũ),
  `EmbeddingRetriever.minimum_score = 0.5` (giá trị khởi điểm, **phải tinh chỉnh thực
  nghiệm** ở bước 7), `HybridRetriever.minimum_score` tính từ hai thành phần.
  `GroundedAnswerEngine.__init__(self, retriever, minimum_score: float | None = None)`:
  `None` → lấy `getattr(retriever, "minimum_score", 0.15)`. Truyền số cụ thể vẫn override
  → backward compatible tuyệt đối.
- **Công thức hybrid (thang tuyệt đối, không normalize theo query)**:
  `score = alpha * cosine + (1 - alpha) * min(lexical_score, 1.0)`, mặc định `alpha = 0.5`.
  Doc nào chỉ xuất hiện ở một bên thì thành phần còn lại = 0 (không loại bỏ doc).
  RRF là phương án thay thế **không chọn** cho lần này vì điểm RRF (~1/61) phá sạch ngữ
  nghĩa ngưỡng `minimum_score`; nếu sau này cần, thêm dưới dạng tham số `fusion=` riêng.
- **Chọn retriever qua factory + env var**: `build_retriever(documents, mode=None, ...)`
  trong `retrieval.py`, `mode` mặc định đọc `os.environ.get("VSF_RETRIEVER", "lexical")`,
  nhận `"lexical" | "embedding" | "hybrid"`. Mode lạ → `ValueError` nêu rõ các giá trị hợp lệ.
- **Mặc định toàn hệ thống = `lexical`.** `api.py` và `scripts/run_eval.py` chuyển sang
  gọi factory, nên hành vi mặc định không đổi nhưng có thể bật embedding bằng env var.

---

## File sẽ đụng tới

- `src/vsf_rag/retrieval.py` — thêm `Retriever` Protocol, `LexicalRetriever.minimum_score`,
  `HybridRetriever`, `build_retriever()`. Giữ nguyên `tokenize`, `Document`,
  `SearchResult`, `load_documents`, thuật toán lexical.
- `src/vsf_rag/retrieval_embedding.py` (MỚI) — `EmbeddingRetriever` + hằng
  `DEFAULT_EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"`.
- `src/vsf_rag/answering.py` — đổi type-hint sang `Retriever`, `minimum_score: float | None = None`.
- `tests/test_embedding_retrieval.py` (MỚI) — test offline (fake encoder) + test so sánh
  hit-rate có `skipif`.
- `tests/test_retrieval.py` — thêm test cho `HybridRetriever` với fake encoder + test
  `build_retriever` factory. KHÔNG sửa 4 test cũ.
- `pyproject.toml` — thêm extra `embeddings = ["sentence-transformers>=3.0"]`.
- `src/vsf_rag/api.py`, `scripts/run_eval.py` — dùng `build_retriever(...)` thay vì
  `LexicalRetriever(...)` trực tiếp.
- `scripts/compare_retrievers.py` (MỚI, tùy chọn) — in bảng so sánh hit-rate 3 chế độ.
- `README.md` — mục cách bật embedding/hybrid + extras.
- `tasks/todo.md` — tick 3 dòng 15-19 của Phase 1.
- KHÔNG sửa: `Dockerfile`, `.github/workflows/ci.yml`, `data/*.jsonl`, `evaluation.py`, `tools.py`.

---

## Các bước cụ thể

1. **`retrieval.py` — thêm Protocol và ngưỡng.**
   - `class Retriever(Protocol)`: `documents: list[Document]`, `minimum_score: float`,
     `def search(self, query: str, top_k: int = 3) -> list[SearchResult]: ...`
   - Thêm class attribute `minimum_score: float = 0.15` cho `LexicalRetriever`
     (có thể override qua `__init__(..., minimum_score: float = 0.15)`).
   - Không đổi công thức tf-idf hiện tại.

2. **`retrieval_embedding.py` — `EmbeddingRetriever`.**
   - `__init__(documents, encoder=None, model_name=DEFAULT_EMBEDDING_MODEL, minimum_score=0.5)`.
   - Rỗng documents → `ValueError("At least one document is required")` (đồng nhất lexical).
   - Encode `f"{doc.title} {doc.text}"` cho từng doc **một lần** trong `__init__`, chuẩn hóa
     L2 ngay; vector 0 → giữ nguyên, không chia cho 0.
   - `search(query, top_k)`: query rỗng / chỉ khoảng trắng → `[]` (khớp lexical:66-71).
     Encode query → normalize → cosine = dot product → sort `(-score, document.id)` để
     tie-break **deterministic** giống lexical:84 → cắt `top_k` (dùng `max(top_k, 1)`).
   - `matched_terms`: điền bằng giao token giữa query và doc (dùng `tokenize` có sẵn) cho
     mục đích giải thích/API; KHÔNG dùng để lọc kết quả (embedding phải trả kết quả cả khi
     không trùng từ nào).
   - Import `sentence_transformers` **trong thân hàm**, bắt `ImportError` và raise lại kèm
     hướng dẫn: `pip install -e '.[embeddings]'`.

3. **`retrieval.py` — `HybridRetriever(lexical, embedding, alpha=0.5, minimum_score=...)`.**
   - Lấy candidate pool = union top-`N` (N = `max(top_k * 4, 10)`) của hai retriever.
   - Điểm theo công thức ở mục Quyết định thiết kế; `matched_terms` lấy từ kết quả lexical
     (rỗng nếu doc chỉ đến từ nhánh embedding).
   - `minimum_score` mặc định = `alpha * emb.minimum_score + (1 - alpha) * min(lex.minimum_score, 1.0)`
     (tính từ hai ngưỡng con, tránh hằng số ma) — vẫn phải verify thực nghiệm ở bước 7.

4. **`retrieval.py` — `build_retriever(documents, mode=None, **kwargs)`.**
   - `mode = mode or os.environ.get("VSF_RETRIEVER", "lexical")`.
   - `"embedding"`/`"hybrid"` mới `from .retrieval_embedding import EmbeddingRetriever`
     (import trong hàm).

5. **`answering.py`** — `def __init__(self, retriever: Retriever, minimum_score: float | None = None)`,
   gán `self.minimum_score = minimum_score if minimum_score is not None else getattr(retriever, "minimum_score", 0.15)`.
   Phần còn lại giữ nguyên tuyệt đối.

6. **`api.py` + `scripts/run_eval.py`** — thay `LexicalRetriever(documents)` bằng
   `build_retriever(documents)`. `/health` thêm field `"retriever": type(engine.retriever).__name__`
   (tiện debug, không phá schema cũ vì chỉ thêm key).

7. **Test — `tests/test_embedding_retrieval.py`.**
   - **Nhóm A (luôn chạy, offline, không cần torch)** — dùng `FakeEncoder` xác định
     (vd: vector đếm tần suất token trên một vocab cố định, hoặc hash-based deterministic):
     - cosine của vector đã normalize nằm trong [-1, 1] và bằng 1.0 khi query trùng hệt doc;
     - thứ tự top-k đúng, tie-break theo `document.id`;
     - query rỗng → `[]`; `top_k=0` → vẫn trả 1 kết quả (khớp lexical);
     - `HybridRetriever` với `alpha=1.0` cho ranking == embedding-only, `alpha=0.0` cho
       ranking == lexical-only (test tính đúng đắn của công thức fusion);
     - `GroundedAnswerEngine` lấy đúng `minimum_score` từ retriever được truyền vào.
   - **Nhóm B (so sánh hit-rate thật, có `skipif`)**:
     - Skip khi `importlib.util.find_spec("sentence_transformers") is None`, và
       skip khi khởi tạo model ném exception (máy CI không có mạng) — dùng
       `pytest.skip(...)` trong fixture `scope="module"` để chỉ tải model **một lần**.
     - Tính hit-rate@3 trên toàn bộ `data/eval.jsonl` (dùng `load_eval_cases` +
       `expected_doc_ids`) cho 3 chế độ, in ra qua `print`/`report` để đọc được với `-s`.
     - Assertion: `embedding_hit_rate >= 0.8` và `hybrid_hit_rate >= max(lexical, embedding)`
       (cho phép sai số nhỏ, vd `>= max(...) - 0.05`). KHÔNG assert
       `embedding > lexical` cứng — eval set 20 case rất nhỏ, dễ flaky.

8. **Tinh chỉnh ngưỡng bằng số liệu thật (BẮT BUỘC, không được đoán).**
   Chạy retriever embedding trên `data/eval.jsonl` + query ngoài miền
   `"Thời tiết hôm nay ở sao Hỏa thế nào?"`, ghi lại phân bố cosine. Chọn
   `EmbeddingRetriever.minimum_score` sao cho: mọi case trong eval set đều `OK`
   (citation coverage = 1.0) và query sao Hỏa ra `UNAVAILABLE`. Cập nhật hằng số mặc định
   theo số đo được, ghi con số quan sát vào `.bangiao/thay-doi.md`.
   Nếu không tồn tại ngưỡng nào thỏa cả hai → ghi rõ trong bàn giao, KHÔNG hạ assertion.

9. **`pyproject.toml`** — thêm:
   ```toml
   [project.optional-dependencies]
   embeddings = ["sentence-transformers>=3.0"]
   ```
   Giữ `dependencies` và `dev` nguyên trạng (KHÔNG đưa sentence-transformers vào `dev`,
   nếu không CI sẽ tải torch mỗi lần chạy).

10. **`scripts/compare_retrievers.py` (tùy chọn)** — in bảng hit-rate/MRR 3 chế độ, hữu ích
    cho README/portfolio. Không thêm vào CI.

11. **`README.md`** — cập nhật sơ đồ Architecture (retriever pluggable), thêm đoạn:
    `pip install -e '.[embeddings]'` và `VSF_RETRIEVER=hybrid uvicorn ...`; nêu rõ mặc
    định là lexical để chạy offline.

12. **`tasks/todo.md`** — tick `[x]` 3 gạch đầu dòng của Phase 1 (dòng 15, 18, 19).

---

## Tiêu chí done

- [ ] `pytest -q` pass **khi chưa cài sentence-transformers** (nhóm B skip, không fail).
- [ ] 4 test cũ trong `tests/test_retrieval.py` pass nguyên trạng, không sửa assertion nào.
- [ ] `python scripts/run_eval.py` (mặc định lexical) cho kết quả **y hệt trước khi sửa**:
      `retrieval_hit_rate >= 0.8`, `citation_coverage == 1.0`.
- [ ] `EmbeddingRetriever.search()` trả `list[SearchResult]` với `score` là cosine
      similarity đã chuẩn hóa, sort giảm dần, tie-break theo `document.id`.
- [ ] `HybridRetriever` với `alpha=1.0` / `alpha=0.0` cho ranking trùng embedding-only /
      lexical-only (có test chứng minh).
- [ ] `build_retriever` hỗ trợ `lexical | embedding | hybrid`, đọc `VSF_RETRIEVER`,
      mặc định `lexical`, mode sai → `ValueError` có thông báo rõ.
- [ ] Không có `import sentence_transformers` ở top-level bất kỳ module nào trong `src/`.
- [ ] Có test so sánh hit-rate lexical vs embedding vs hybrid trên `data/eval.jsonl`,
      in được số liệu 3 chế độ.
- [ ] Đã chạy thực tế với model thật ít nhất 1 lần và ghi số hit-rate 3 chế độ +
      ngưỡng `minimum_score` chọn được vào `.bangiao/thay-doi.md`.
- [ ] Query "Thời tiết hôm nay ở sao Hỏa thế nào?" vẫn ra `UNAVAILABLE` ở **cả 3** chế độ.
- [ ] `pyproject.toml` có extra `embeddings`; `dev` và `dependencies` không đổi.
- [ ] `Dockerfile` và `.github/workflows/ci.yml` không bị sửa.
- [ ] `tasks/todo.md` Phase 1 đã tick.
