"""Thin wrappers so the chatbot works with any pretrained LLM.

  AnthropicLLM   - Claude via the Anthropic API            (env ANTHROPIC_API_KEY)
  OpenAICompatLLM- OpenAI, Ollama, vLLM, LM Studio, etc.   (free/local with Ollama)
  None           - offline mode; the chatbot falls back to a keyword router
"""
from __future__ import annotations

import os


class LLMError(RuntimeError):
    pass


class AnthropicLLM:
    def __init__(self, model: str = "claude-sonnet-5-5", api_key: str | None = None):
        try:
            import anthropic
        except ImportError as e:
            raise LLMError("pip install anthropic") from e
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def complete(self, system: str, messages: list[dict], max_tokens: int = 1200) -> str:
        try:
            r = self.client.messages.create(model=self.model, system=system, messages=messages,
                                            max_tokens=max_tokens, temperature=0)
            return "".join(b.text for b in r.content if b.type == "text")
        except Exception as e:
            raise LLMError(str(e)) from e


class OpenAICompatLLM:
    """Works with any /v1/chat/completions server. Ollama default: http://localhost:11434/v1"""

    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434/v1",
                 api_key: str | None = None):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "ollama")

    def complete(self, system: str, messages: list[dict], max_tokens: int = 1200) -> str:
        import requests
        try:
            r = requests.post(f"{self.base_url}/chat/completions", timeout=120,
                              headers={"Authorization": f"Bearer {self.api_key}"},
                              json={"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                                    "messages": [{"role": "system", "content": system}, *messages]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise LLMError(str(e)) from e
