import os
from pathlib import Path

def configure_hf_env(project_root: Path) -> Path:
    """
    Configure HuggingFace cache and networking behavior.
    Must be executed before importing transformers/sentence_transformers.
    """
    models_dir = project_root / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Use HF_HOME as the canonical cache root (TRANSFORMERS_CACHE is deprecated).
    os.environ.setdefault("HF_HOME", str(models_dir))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    # Optional: use a mirror endpoint if huggingface.co is slow/unreachable.
    # You can also set this in PowerShell: $env:HF_ENDPOINT="https://hf-mirror.com"
    os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

    # Increase timeouts to avoid default 10s connect timeout in some environments.
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

    # Reduce irrelevant TF logs (best effort; if TF is installed it may still print).
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return models_dir

def configure_hf_cache(project_root: Path) -> Path:
    """
    Configure HuggingFace cache folders under project directory.
    Must be called before importing transformers/sentence_transformers.
    """
    models_dir = project_root / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(models_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(models_dir))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return models_dir

def prepare_assets() -> None:
    """
    Prepare all heavy assets on disk and warm up heavy models once.

    This function is safe to call multiple times.
    It will only build/load when assets are missing.
    """
    project_root = Path(__file__).resolve().parents[1]
    configure_hf_cache(project_root)
    # Optional: silence HF progress bars if desired
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    # Imports must happen after env vars are set
    from rag.config import RagConfig
    from rag.pipeline.stores import load_bm25_store, load_semantic_store
    from scripts.build_corpus_from_mds import main as build_corpus_main

    cfg = RagConfig()

    # Build corpus if missing
    corpus_path = Path(cfg.corpus_jsonl)
    if not corpus_path.exists():
        build_corpus_main()

    # Build/load BM25 and semantic stores (they build indexes if missing)
    load_bm25_store(cfg)
    load_semantic_store(cfg)

    # Warm up ColBERT reranker model if enabled (uses local cache_dir set above)
    if getattr(cfg, "use_colbert_rerank", False):
        from rag.rerank.colbert_rerank import _load_model_and_tokenizer
        _load_model_and_tokenizer(getattr(cfg, "colbert_model_name", "distilbert-base-uncased"))

if __name__ == "__main__":
    prepare_assets()
    print("Assets prepared successfully.")
