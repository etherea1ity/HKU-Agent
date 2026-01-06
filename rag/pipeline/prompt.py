from rag.config import RagConfig


SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant. Answer the user's question using ONLY the provided context.\n"
    "If the context is insufficient, say you don't know.\n"
    "When you use evidence, cite it using [1], [2], ... matching the context blocks.\n"
)


def make_prompt(cfg: RagConfig, question: str, context: str) -> str:
    return (
        f"{SYSTEM_INSTRUCTIONS}\n"
        f"[Question]\n{question.strip()}\n\n"
        f"[Context]\n{context.strip()}\n\n"
        f"[Answer]\n"
    )
