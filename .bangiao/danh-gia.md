Đã kiểm tra `git status --short` và `git diff HEAD`; hiện có thay đổi ngoài phạm vi kế hoạch gốc trong `HEAD:.bangiao/ke-hoach.md` (Phase 0). Kế hoạch gốc chỉ cho phép data, test và tasks/todo.md, nhưng diff hiện tại sửa `.bangiao/*`, `.gitignore`, README, pyproject, scripts, toàn bộ `src/` và thêm file mới; đồng thời không có thay đổi data. `.bangiao/ke-hoach.md:1` bị rewrite sang Phase 1, nên không thể dùng bản đã sửa để thay thế kế hoạch gốc mà không có xác nhận/tách task riêng.

Nếu chấp nhận Phase 1 là phạm vi mới, phần code chính nhìn chung đúng: import embedding lazy, dependency optional, factory mặc định lexical, regression lexical và `run_eval` mặc định đều ổn. Tuy nhiên vẫn có các điểm cần sửa:

- `src/vsf_rag/retrieval.py:169-171` dùng toàn bộ `len(self.documents)` làm candidate pool, không đúng thiết kế `N=max(top_k*4,10)` tại `.bangiao/ke-hoach.md:121-126`; gây chi phí và phạm vi fusion lớn hơn cam kết, dù hiện dữ liệu chỉ có 29 document.
- `tests/test_embedding_retrieval.py:287-306` vẫn bắt `OSError` rất rộng rồi `pytest.skip`; lỗi model/library hoặc lỗi tích hợp cũng có thể bị che thành môi trường thiếu model. Nên chỉ bắt các lỗi tải model/network đã xác định.
- Test factory chỉ kiểm tra mode mặc định và mode sai tại `tests/test_embedding_retrieval.py:271-278`, chưa kiểm tra thực tế `VSF_RETRIEVER=embedding|hybrid`, truyền kwargs và hành vi lazy dependency.
- Tiêu chí `.bangiao/ke-hoach.md:202-203` yêu cầu ghi hit-rate của cả 3 retriever vào `.bangiao/thay-doi.md`, nhưng `.bangiao/thay-doi.md:17-19` chỉ ghi kết quả lexical; không có số embedding/hybrid thực tế.
- `.bangiao/ket-qua-test.md:12` khai báo `21 passed, 0 skipped`, nhưng chạy độc lập hiện tại với `python3 -m pytest -q -s -rs` cho `19 passed, 2 skipped`; hai test real-model bị skip vì môi trường không có thư mục tạm khả dụng. Lệnh `python3 -m pytest -q` còn thất bại ngay ở pytest capture vì cùng nguyên nhân. Vì vậy chưa thể xác nhận claim test/model thật trong artifact bàn giao.
- `.gitignore:9` thêm `.bangiao/`, làm các artifact bàn giao mới trong thư mục này dễ bị bỏ qua khỏi Git; đây là rủi ro quy trình và không thuộc phạm vi Phase 1.
- Có rủi ro vận hành mức vừa: bật `embedding`/`hybrid` khiến app tải model từ mạng ngay lúc import/startup; model và dependency chỉ pin theo lower bound, nên có nguy cơ startup fail hoặc thay đổi hành vi theo môi trường. Không phát hiện lỗ hổng bảo mật nghiêm trọng mức CHAN.

Kết quả xác nhận được: `python3 scripts/run_eval.py` exit 0 với 20 case, retrieval/citation/keyword đều 1.0; các test offline pass. Cần tách/khôi phục kế hoạch gốc hoặc được phê duyệt phạm vi Phase 1, cập nhật bằng chứng test đầy đủ và xử lý các điểm trên trước khi commit.

PHAN QUYET: CAN SUA
