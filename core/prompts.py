from typing import Dict, List


def make_rewrite_messages(user_question: str) -> List[Dict[str, str]]:
    """
    Rewrite user question into a retrieval-friendly query.
    Output should be short and keyword/entity focused.
    """
    system = (
        "You are a query rewriting assistant for a RAG system.\n"
        "Rewrite the user's question into a search query.\n"
        "Rules:\n"
        "- Keep it short (<= 20 words if possible)\n"
        "- Keep course codes / numbers / key entities\n"
        "- Remove filler words\n"
        "- Output ONLY the rewritten query, no extra text\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_question.strip()},
    ]


def make_answer_messages(user_question: str, context: str) -> List[Dict[str, str]]:
    """
    Answer with citations like [1], [2] based on provided context blocks.
    """
    system = (
        "You are a helpful assistant.\n"
        "Answer the user's question using ONLY the provided context.\n"
        "If the context is insufficient, say you don't know.\n"
        "Cite evidence using [1], [2], ... matching the context blocks.\n"
    )
    user = (
        f"[Question]\n{user_question.strip()}\n\n"
        f"[Context]\n{context.strip()}\n\n"
        f"[Answer]\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
