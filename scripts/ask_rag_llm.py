import os
import sys

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.config import RagConfig
from core.llm_client import LLMClient, LLMConfig
from agent.flow_rag import RagFlow


def main():
    cfg = RagConfig(use_colbert_rerank=True)
    llm = LLMClient(LLMConfig())
    flow = RagFlow(cfg=cfg, llm=llm)

    while True:
        q = input("\nUser> ").strip()
        if not q:
            break

        out = flow.run(q)

        print("\n===== Rewrite =====")
        print(out["rewritten_query"])

        print("\n===== Answer =====")
        print(out["answer"])

        print("\n===== Latency / Usage =====")
        print("latency_ms:", out["latency_ms"])
        print("usage.rewrite:", out["usage"]["rewrite"])
        print("usage.answer:", out["usage"]["answer"])


if __name__ == "__main__":
    main()
