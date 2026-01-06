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

def prepare_assets() -> None:
    project_root = Path(__file__).resolve().parents[1]
    configure_hf_env(project_root)

    # Import after env vars are set
    from rag.config import RagConfig
    from rag.pipeline.stores import load_bm25_store, load_semantic_store
    from scripts.build_corpus_from_mds import main as build_corpus_main

    cfg = RagConfig()

    # 1) Ensure corpus exists
    corpus_path = Path(cfg.corpus_jsonl)
    if not corpus_path.exists():
        build_corpus_main()

    # 2) Ensure BM25 exists (does not require HF)
    load_bm25_store(cfg)

    # 3) Ensure semantic index exists (requires HF model download at least once)
    try:
        load_semantic_store(cfg)
    except Exception as e:
        # Do not fail the whole preparation; allow BM25-only mode temporarily.
        print("[prepare_assets] Semantic store preparation failed.")
        print("Reason:", repr(e))
        print("You can still run BM25-only. Fix network/HF_ENDPOINT and rerun prepare_assets later.")

    print("prepare_assets done.")

if __name__ == "__main__":
    prepare_assets()
