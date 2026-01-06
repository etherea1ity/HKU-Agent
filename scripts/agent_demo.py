import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.config import RagConfig
from agent.agent import RagAgent
from agent.types import AgentState


def main():
    cfg = RagConfig()
    agent = RagAgent(cfg=cfg)
    state = AgentState()

    print("Agent demo. Type empty line to exit.")

    while True:
        user_text = input("\nUser> ").strip()
        if not user_text:
            break

        result = agent.run(state, user_text)

        print("\n===== TRACE =====")
        for e in result.events:
            print(f"- {e.name} | {e.attrs}")

        print("\n===== PROMPT (preview) =====\n")
        print(result.prompt)

        print("\n===== CONTEXT (preview) =====\n")
        print(result.context[:1200] + ("..." if len(result.context) > 1200 else ""))


if __name__ == "__main__":
    main()
