# Plan: Nâng cấp thành "Vietnamese RAG + Agent Evaluation Harness"

Mục tiêu: định vị lại project để khớp JD VinSmart Future - AI Engineer
(NLP/RAG, generative AI, agentic systems, ML Ops, production API).

Nguồn JD: `/Users/nguyenvokhang/Downloads/VSF_JD AI Engineer_HCM.docx`

## Phase 0 — Chốt phạm vi
- [x] LLM provider: **OpenRouter với model free** (OpenAI-compatible API, base_url riêng, cần OPENROUTER_API_KEY)
- [x] Embedding model tiếng Việt: **bkai-foundation-models/vietnamese-bi-encoder** (qua sentence-transformers)
- [x] Công cụ tracking: **MLflow (self-host)**
- [x] Mở rộng `data/knowledge_base.jsonl` và `data/eval.jsonl` (hiện chỉ 5 dòng — cần đủ đa dạng để số liệu eval có ý nghĩa)

## Phase 1 — Embedding retrieval (thay lexical-only)
- [x] Thêm `EmbeddingRetriever` trong `src/vsf_rag/retrieval.py` (hoặc file mới `retrieval_embedding.py`)
  - Input: query string, danh sách document chunks
  - Output: top-k chunks kèm cosine similarity score
- [x] Giữ retriever lexical (BM25) hiện tại làm fallback/so sánh — implement **hybrid** (kết hợp điểm) hoặc cho phép chọn qua config
- [x] Unit test: so sánh hit-rate lexical vs embedding trên `data/eval.jsonl`

## Phase 2 — LLM-based generation (thay extractive rule-based)
- [x] Thiết kế `LLMProvider` interface (`generate(query, context) -> LLMAnswer`) trong `src/vsf_rag/llm.py`
  - Input: query + retrieved context (list `ContextChunk` với citation id)
  - Output: câu trả lời tiếng Việt + danh sách citation id được dùng
- [x] Implement `OpenRouterProvider` (API-backed, OpenAI-compatible `/chat/completions`) theo lựa chọn ở Phase 0.
      Model mặc định khi không set `OPENROUTER_MODEL`: `openrouter/free` (auto-router free chính thức của OpenRouter).
- [x] Thêm prompt template (`SYSTEM_PROMPT_VI`) ép model trích dẫn nguồn và trả lời "không đủ dữ liệu" khi context rỗng/không liên quan
- [x] Cập nhật `answering.py` (`build_answer_engine`) để gọi qua interface này thay vì rule-based cũ (giữ rule-based làm fallback tự động khi thiếu `OPENROUTER_API_KEY` hoặc khi `LLMError` — hữu ích cho CI/demo offline)

## Phase 3 — Agent / tool-use tối thiểu
- [x] Định nghĩa 1-2 tool thật trong `tools.py` (vd: tính toán, tra cứu ngày giờ, hoặc gọi 1 API public đơn giản)
  - Input: câu hỏi được router phân loại là "cần tool"
  - Output: kết quả tool + log trace (tool nào được gọi, tham số gì, kết quả gì)
- [x] Router: LLM hoặc rule đơn giản quyết định dùng RAG hay tool
- [x] Log trace ra response để minh bạch quá trình quyết định (phục vụ demo/portfolio)

## Phase 4 — Evaluation nâng cấp (RAGAS-style)
- [x] Retrieval: hit-rate, MRR@k
- [x] Generation: faithfulness (câu trả lời có bám context không), answer relevancy
- [x] Citation coverage (giữ từ bản cũ)
- [x] Latency per stage (retrieval / generation / total)
- [x] Xuất kết quả eval ra file (json/csv) + log vào MLflow/W&B đã chọn ở Phase 0

## Phase 5 — API, Docker, CI (giữ + hoàn thiện)
- [x] Đảm bảo `/query`, `/evaluate` hoạt động với pipeline mới, cập nhật OpenAPI examples
- [x] Cập nhật `Dockerfile` nếu thêm dependency mới (torch/sentence-transformers nặng — cân nhắc image size)
- [x] Thêm GitHub Actions chạy pytest + eval trên mỗi PR (roadmap mục 6 trong README cũ)
- [x] Cập nhật `README.md`: kiến trúc mới, cách chạy, số liệu eval mẫu

## Phase 6 — Polish cho CV/portfolio
- [x] Viết 1 đoạn mô tả ngắn (README) nêu rõ: vấn đề giải quyết, kiến trúc, số liệu eval đạt được
- [x] Chụp/ghi demo ngắn hoặc GIF cho README
- [x] Đối chiếu lại với từng gạch đầu dòng JD, đảm bảo README nêu rõ ánh xạ (giúp khi phỏng vấn dễ trình bày)

## Review (điền sau khi hoàn thành)
- Kết quả đạt được: Phase 5 hoàn thành — `POST /evaluate` chạy được harness Phase 4 qua HTTP
  (mặc định 20 case checked-in, hỗ trợ `cases` tuỳ chỉnh, `include_details` để ẩn/hiện chi tiết
  từng case); OpenAPI có `openapi_tags` + example cho `QueryRequest`/`ToolRequest`/`EvaluateRequest`
  và response models (`QueryResponse`/`HealthResponse`/`ToolsResponse`/`ToolCallResponse`) khớp
  100% field với `Answer.to_dict()`; `Dockerfile` single-stage non-root có `HEALTHCHECK` và
  `ARG EXTRAS` cho biến thể embeddings; `.github/workflows/ci.yml` chạy matrix Python 3.10/3.11 +
  eval offline + `scripts/check_eval_gate.py` (quality gate) + upload artifact + job `docker`
  build-only; README cập nhật badge CI, mục "Evaluation API", "Docker", "Continuous Integration".
- Số liệu eval trước/sau: không đổi so với Phase 4 (logic `evaluate()` không sửa) — lexical +
  extractive trên 20 case: `retrieval_hit_rate=1.0`, `retrieval_mrr_at_k=1.0`,
  `citation_coverage=1.0`, `faithfulness≈0.94`, `answer_relevancy≈0.61`.
- Điểm còn thiếu so với JD (nếu có): chưa có demo deployment thật (chỉ Docker build-only trong
  CI, không push image/deploy); `docker build`/`docker run` chưa verify được trong môi trường
  thực thi task này vì Docker daemon không chạy sẵn ở sandbox — cần verify thủ công trước khi
  release.

- Phase 6 hoàn thành: README thêm elevator pitch + 3 badge mới (Python, MIT
  License, Docker) bên cạnh badge CI; sơ đồ kiến trúc ASCII cũ thay bằng
  flowchart Mermaid (render ảnh thật trên GitHub, không cần asset binary
  riêng); thêm `docs/demo.gif` — GIF terminal ghi lại 2 request `/query` thật
  chạy trên service đang chạy local (route=tool cho câu tính toán, route=rag
  kèm citation cho câu hỏi tiếng Việt), dựng bằng Pillow từ response JSON
  thật (không phải dữ liệu giả) vì Chrome extension (`gif_creator`) không kết
  nối được trong môi trường này và các tool ghi terminal (`asciinema`, `vhs`,
  `agg`) không có sẵn; thêm section "JD → project mapping" (bảng 7 dòng ánh
  xạ từng gạch đầu dòng JD AI Engineer @ VinSmart Future tới file/section cụ
  thể trong repo).
- Điểm còn thiếu: GIF là mô phỏng terminal render bằng Pillow (không phải
  screen-recording thật của trình duyệt/Swagger UI) do giới hạn môi trường —
  nội dung/response bên trong là thật 100%, chỉ phần trình bày là dựng lại;
  nên cân nhắc quay lại bằng Chrome/asciinema khi có môi trường phù hợp nếu
  muốn một bản ghi màn hình thực sự.
