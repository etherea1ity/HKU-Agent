from dataclasses import dataclass


@dataclass(frozen=True)
class RagConfig:
    # Fusion mode: 
    # - "rrf": existing Reciprocal Rank Fusion (default)
    # - "lr": learned fusion using a trained Logistic Regression model
    fusion_mode: str = "rrf"
    learned_fusion_model_path: str = "rag/models/learned_fusion_lr.joblib"

    # ColBERT-style rerank (late interaction)
    use_colbert_rerank: bool = False
    rerank_candidates_k: int = 50  # how many candidates to rerank after hybrid recall

    # IMPORTANT:
    # Use Hugging Face repo_id here (NOT a Hugging Face cache folder like models--xxx).
    # Cache location is controlled by environment variables in app/main.py.
    colbert_model_name: str = "distilbert-base-uncased"
    colbert_query_max_length: int = 64
    colbert_doc_max_length: int = 256
    colbert_normalize: bool = True

    # Corpus / index paths
    corpus_jsonl: str = "data/corpus/corpus.jsonl"
    bm25_index_path: str = "data/index/bm25.pkl"
    semantic_index_dir: str = "data/index/semantic"

    # Retrieval
    bm25_k: int = 20
    sem_k: int = 20
    rrf_k: int = 60
    sem_ef_search: int = 128  # HNSW query-time breadth

    # Retrieval sizing
    recall_k: int = 120
    final_k: int = 12

    # Thresholding
    score_ratio: float = 0.65
    score_abs_threshold: float = float("-inf")
    min_keep: int = 5
    max_keep: int = 12

    # Rerank
    rerank_k: int = 30

    # Debug
    debug_retrieve: bool = False

    # Context formatting
    max_chunks_in_context: int = 6
    max_chars_per_chunk: int = 900
    max_total_context_chars: int = 4500

    # IMPORTANT:
    # Use repo_id here as well. SentenceTransformer will use HF cache automatically.
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Semantic index type: "hnsw" or "flat"
    semantic_index_type: str = "hnsw"

    # HNSW build params (only used when building semantic index)
    hnsw_m: int = 32
    ef_construction: int = 200
    ef_search_default: int = 64
