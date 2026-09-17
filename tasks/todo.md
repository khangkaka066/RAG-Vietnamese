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
- [ ] Thiết kế `LLMProvider` interface (`generate(prompt, context) -> answer`) trong `src/vsf_rag/llm.py`
  - Input: query + retrieved context (list chunk + citation id)
  - Output: câu trả lời tiếng Việt + danh sách citation id được dùng
- [ ] Implement 1 provider cụ thể (API-backed) theo lựa chọn ở Phase 0
- [ ] Thêm prompt template ép model trích dẫn nguồn và trả lời "không đủ dữ liệu" khi context rỗng/không liên quan
- [ ] Cập nhật `answering.py` để gọi qua interface này thay vì rule-based cũ (giữ rule-based làm fallback không cần API key — hữu ích cho CI/demo offline)

## Phase 3 — Agent / tool-use tối thiểu
- [ ] Định nghĩa 1-2 tool thật trong `tools.py` (vd: tính toán, tra cứu ngày giờ, hoặc gọi 1 API public đơn giản)
  - Input: câu hỏi được router phân loại là "cần tool"
  - Output: kết quả tool + log trace (tool nào được gọi, tham số gì, kết quả gì)
- [ ] Router: LLM hoặc rule đơn giản quyết định dùng RAG hay tool
- [ ] Log trace ra response để minh bạch quá trình quyết định (phục vụ demo/portfolio)

## Phase 4 — Evaluation nâng cấp (RAGAS-style)
- [ ] Retrieval: hit-rate, MRR@k
- [ ] Generation: faithfulness (câu trả lời có bám context không), answer relevancy
- [ ] Citation coverage (giữ từ bản cũ)
- [ ] Latency per stage (retrieval / generation / total)
- [ ] Xuất kết quả eval ra file (json/csv) + log vào MLflow/W&B đã chọn ở Phase 0

## Phase 5 — API, Docker, CI (giữ + hoàn thiện)
- [ ] Đảm bảo `/query`, `/evaluate` hoạt động với pipeline mới, cập nhật OpenAPI examples
- [ ] Cập nhật `Dockerfile` nếu thêm dependency mới (torch/sentence-transformers nặng — cân nhắc image size)
- [ ] Thêm GitHub Actions chạy pytest + eval trên mỗi PR (roadmap mục 6 trong README cũ)
- [ ] Cập nhật `README.md`: kiến trúc mới, cách chạy, số liệu eval mẫu

## Phase 6 — Polish cho CV/portfolio
- [ ] Viết 1 đoạn mô tả ngắn (README) nêu rõ: vấn đề giải quyết, kiến trúc, số liệu eval đạt được
- [ ] Chụp/ghi demo ngắn hoặc GIF cho README
- [ ] Đối chiếu lại với từng gạch đầu dòng JD, đảm bảo README nêu rõ ánh xạ (giúp khi phỏng vấn dễ trình bày)

## Review (điền sau khi hoàn thành)
- Kết quả đạt được:
- Số liệu eval trước/sau:
- Điểm còn thiếu so với JD (nếu có):
