import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI


@dataclass
class LLMConfig:
    api_key_env: str = "sk-4c00b82305614b50a3d5d5c68b815376"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "deepseek-v3.2"
    enable_thinking: bool = False
    timeout: float = 120.0


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        # api_key = os.getenv(cfg.api_key_env)
        # if not api_key:
        #     raise RuntimeError(
        #         f"Missing API key env var: {cfg.api_key_env}. "
        #         f"Set it in PowerShell: $env:{cfg.api_key_env}='...'"
        #     )

        self.cfg = cfg
        self.client = OpenAI(api_key="sk-4c00b82305614b50a3d5d5c68b815376", base_url=cfg.base_url)

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        enable_thinking: Optional[bool] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Non-streaming chat completion.
        Returns (answer_text, usage_dict).
        """
        if enable_thinking is None:
            enable_thinking = self.cfg.enable_thinking

        eb = dict(extra_body or {})
        eb.setdefault("enable_thinking", bool(enable_thinking))

        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            stream=False,
            extra_body=eb,
            timeout=self.cfg.timeout,
        )

        text = resp.choices[0].message.content or ""
        usage = resp.usage.model_dump() if resp.usage else {}
        return text, usage

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        *,
        enable_thinking: Optional[bool] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Iterable[Dict[str, Any]]:
        """
        Streaming chat completion generator.
        Yields dict events:
          {"type": "reasoning", "text": "..."}
          {"type": "content", "text": "..."}
          {"type": "usage", "usage": {...}}
        """
        if enable_thinking is None:
            enable_thinking = self.cfg.enable_thinking

        eb = dict(extra_body or {})
        eb.setdefault("enable_thinking", bool(enable_thinking))

        stream = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=eb,
            timeout=self.cfg.timeout,
        )

        for chunk in stream:
            # usage frame
            if not getattr(chunk, "choices", None):
                usage = chunk.usage.model_dump() if getattr(chunk, "usage", None) else {}
                yield {"type": "usage", "usage": usage}
                continue

            delta = chunk.choices[0].delta

            # reasoning_content (DashScope compatible fields)
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                yield {"type": "reasoning", "text": rc}

            # normal content
            c = getattr(delta, "content", None)
            if c:
                yield {"type": "content", "text": c}
