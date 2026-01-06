from dataclasses import dataclass


@dataclass(frozen=True)
class RagConfig:
    # ColBERT-style rerank (late interaction)
    use_colbert_rerank: bool = False
    rerank_candidates_k: int = 50  # how many candidates to rerank after hybrid recall

    colbert_model_name: str = "distilbert-base-uncased"
    colbert_query_max_length: int = 64
    colbert_doc_max_length: int = 256
    colbert_normalize: bool = True

    # Corpus / index paths
    corpus_jsonl: str = "data/corpus/corpus.jsonl"
    bm25_index_path: str = "data/index/bm25.pkl"
    semantic_index_dir: str = "data/index/semantic"

    # Retrieval
    top_k: int = 5
    bm25_k: int = 20
    sem_k: int = 20
    rrf_k: int = 60
    sem_ef_search: int = 128  # HNSW query-time breadth

    # Context formatting
    max_chunks_in_context: int = 6
    max_chars_per_chunk: int = 900
    max_total_context_chars: int = 4500

    # Semantic embedding model (only used when semantic index is built)
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Semantic index type: "hnsw" or "flat"
    semantic_index_type: str = "hnsw"

    # HNSW build params (only used when building semantic index)
    hnsw_m: int = 32
    ef_construction: int = 200
    ef_search_default: int = 64
