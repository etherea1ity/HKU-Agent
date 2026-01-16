import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

@dataclass
class LLMConfig:
    # Read API key from this env var name
    api_key_env: str = "DASHSCOPE_API_KEY"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "deepseek-v3.2"
    timeout: float = 120.0


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        # Prefer env var. If cfg.api_key_env looks like a key itself, allow it as a fallback.
        api_key = os.getenv(cfg.api_key_env)
        if not api_key and (cfg.api_key_env or "").startswith("sk-"):
            api_key = cfg.api_key_env
        if not api_key:
            raise RuntimeError(
                f"Missing API key env var: {cfg.api_key_env}. "
                f"Set it before starting the server."
            )

        self.client = OpenAI(api_key=api_key, base_url=cfg.base_url)

    def chat(self, messages: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        Non-stream chat call. Returns (text, usage_dict).
        """
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            timeout=self.cfg.timeout,
        )
        text = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        return text, (usage.model_dump() if usage else {})

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Iterable[Dict[str, Any]]:
        """
        Streaming chat call. Yields events:
          - {"type": "delta", "text": "..."}
          - {"type": "usage", "usage": {...}}  (when available)
          - {"type": "done"}
        """
        stream = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=extra_body or {},
            timeout=self.cfg.timeout,
        )

        for chunk in stream:
            # Delta tokens
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield {"type": "delta", "text": chunk.choices[0].delta.content}

            # Usage (some providers send it on the final chunk)
            if getattr(chunk, "usage", None):
                yield {"type": "usage", "usage": chunk.usage.model_dump()}

        yield {"type": "done"}
