# Vietnamese Legal RAG Pipeline

Hệ thống Retrieval-Augmented Generation (RAG) hỏi-đáp pháp luật Việt Nam, kết hợp tìm kiếm dense + sparse (hybrid search), reranking, và sinh câu trả lời bằng LLM, có kèm bộ đánh giá (ablation study, ROUGE-L, độ chính xác trích dẫn).

## Cấu trúc dự án

| File | Mô tả |
|---|---|
| `corpus_building.ipynb` | Xây dựng kho tri thức (corpus) pháp luật từ dữ liệu thô |
| `RAG_pipeline.ipynb` | Indexing, retrieval (dense/sparse/hybrid), generation và đánh giá hệ thống RAG |

## 1. `corpus_building.ipynb` — Xây dựng corpus

- **Nguồn dữ liệu:** dataset `adamwhite625/vietnam-legal-qa` (`ready_to_import_dataset.json`) tải từ Hugging Face Hub.
**Đầu ra:** `corpus.json` — kho tri thức pháp luật đã làm sạch, dùng làm đầu vào cho bước indexing ở `RAG_pipeline.ipynb`.

## 2. `RAG_pipeline.ipynb` — Pipeline RAG

### Dữ liệu đầu vào
Tải từ repo Hugging Face `l3mon3/embedded_vector`:
- `corpus.json` — kho tri thức (từ bước 1)
- `embeddings.npy` — vector embedding của corpus
- `query_embeddings.npy` — vector embedding của tập câu hỏi test
- `test-00000-of-00001.parquet` — tập câu hỏi/đáp án chuẩn để đánh giá

### Indexing
- **Dense index:** encode văn bản bằng `SentenceTransformer('BAAI/bge-m3')`, lưu vào **ChromaDB** (`PersistentClient`, collection `legal_corpus`, không gian cosine).
- **Sparse index:** tokenize văn bản và xây chỉ mục **BM25** (`rank_bm25.BM25Okapi`), lưu ra file pickle `bm25`.

### Chuẩn bị tập test
- Ánh xạ mỗi câu hỏi test sang `target_doc_id` tương ứng trong corpus (dựa trên `law_name` + `law_id`).
- Trích xuất câu hỏi (`query_to_embed`) và câu trả lời chuẩn (`ground_truth_ans`) từ trường `messages` (định dạng hội thoại user/assistant).

### Retrieval
- **Dense search:** truy vấn ChromaDB, trả về điểm tương đồng cosine.
- **Sparse search:** truy vấn BM25.
- **Hybrid search (`hybrid_search_ablation`):** kết hợp kết quả dense + sparse bằng **Reciprocal Rank Fusion (RRF)**.
- **Reranking:** `CrossEncoder('BAAI/bge-reranker-v2-m3')` để sắp xếp lại top ứng viên sau truy vấn.

### Generation
- **LLM:** `Qwen/Qwen2.5-3B-Instruct`.

### Đánh giá (Evaluation)
- **`check_citation_accuracy`:** kiểm tra câu trả lời có chứa đúng tên luật và số điều luật chuẩn hay không.
- **`check_format_adherence`:** kiểm tra câu trả lời có tuân thủ định dạng yêu cầu (ví dụ mục "Căn cứ pháp lý", "Nội dung", ...).
- **ROUGE-L:** đo độ tương đồng giữa câu trả lời sinh ra và câu trả lời chuẩn (`rouge_score`).
- **`run_custom_evaluation`:** đánh giá toàn bộ tập test theo citation accuracy, format adherence, ROUGE-L; xuất báo cáo CSV.
- **`run_ablation_study` / `evaluate_retrieval_ablation`:** so sánh hiệu năng giữa 4 cấu hình:
  - `dense_only`
  - `sparse_only`
  - `hybrid_no_rerank`
  - `hybrid_with_rerank`
  
  Đánh giá retrieval theo nhiều `top_k` (1, 3, 5).
- **`run_zero_shot_evaluation`:** đánh giá LLM trả lời trực tiếp (không RAG) làm baseline so sánh.

## Yêu cầu cài đặt

```bash
pip install huggingface_hub pandas numpy python-slugify
pip install chromadb sentence-transformers rank-bm25
pip install rouge-score nltk transformers accelerate bitsandbytes torch
```

## Quy trình chạy

1. Chạy `corpus_building.ipynb` để tạo `corpus.json` từ dữ liệu thô (hoặc dùng corpus đã có sẵn trên Hugging Face).
2. Chạy `RAG_pipeline.ipynb` để:
   - Tải corpus + embeddings đã tính sẵn.
   - Dựng dense index (ChromaDB) và sparse index (BM25).
   - Thực hiện hybrid retrieval + rerank.
   - Sinh câu trả lời bằng LLM.
   - Chạy các bước đánh giá / ablation study và xuất báo cáo CSV.


## Hardware require:
Please run both on Kaggle or Colab with GPU turning on !

## Ghi chú

- Một số cell trong `RAG_pipeline.ipynb` (encode embeddings, chạy ablation study đầy đủ) đang bị comment vì tốn tài nguyên/thời gian — dùng embeddings đã tính sẵn tải từ Hugging Face để chạy nhanh.
- Notebook sử dụng GPU (`cuda:1`, `device_map={'': 0}`) cho reranker và LLM — cần môi trường có GPU để chạy generation.