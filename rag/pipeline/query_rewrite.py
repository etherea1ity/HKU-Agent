def rewrite_query(user_question: str) -> str:
    """
    Minimal version: no rewrite.
    Later you can add:
      - keyword extraction
      - multi-query expansion
      - query rewrite via LLM
    """
    return user_question.strip()
