# scripts/prefetch_models.py
from __future__ import annotations

from pathlib import Path
from huggingface_hub import snapshot_download


def prefetch(repo_id: str, target_dir: Path) -> None:
    """
    Download a Hugging Face model snapshot into a local directory.
    This is a one-time operation; subsequent runs will reuse the local files.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    print(f"[OK] Prefetched {repo_id} -> {target_dir}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    models_root = project_root / "data" / "models" / "hf"

    # Your RagConfig defaults
    embedding_repo = "sentence-transformers/all-MiniLM-L6-v2"
    colbert_repo = "distilbert-base-uncased"

    prefetch(embedding_repo, models_root / "embedding_all-MiniLM-L6-v2")
    prefetch(colbert_repo, models_root / "colbert_distilbert-base-uncased")


if __name__ == "__main__":
    main()
