"""NVIDIA's hosted API (build.nvidia.com) as a structured-output chain.

Owner decision (2026-10-23, docs/model-selection-extraction.md and docs/decisions.md): extraction
falls back to ``nvidia/nemotron-3.5-super-vl-120b-a12b`` when Gemini's daily budget is spent. The
owner accepted NVIDIA's API trial terms for this use.

The endpoint is OpenAI-compatible; it is called over plain HTTP (``httpx``), with the output forced
to the caller's Pydantic schema (``response_format: json_schema``) and the model's thinking switched
off. The object exposes the same ``invoke(prompt)`` as the Gemini chain, so the extraction loop,
its counter, its guards and its failure classes are shared:

* 429, 5xx, a timeout or a dropped connection -> ``HTTPProviderTransient`` (retried by the caller,
  with the owner's jittered backoff, honouring Retry-After);
* any other 4xx -> ``HTTPProviderRefused`` (an ``api_error``: the run stops);
* an answer that isn't the schema -> ``pydantic.ValidationError``; an empty one -> ``None``
  (both are schema errors to the caller).

The key comes from ``NVIDIA_API_KEY`` in ``.env`` and is never logged or put in an error message.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

from hiredra.quota import HTTPProviderRefused, HTTPProviderTransient

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_OBJECT_RE = re.compile(r"\{.*\}", re.S)


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers.get("retry-after", ""))
    except ValueError:
        return None


class NvidiaChain:
    """``invoke(prompt) -> schema instance | None`` against one NVIDIA hosted model."""

    def __init__(
        self,
        model: str,
        schema: type[BaseModel],
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        max_tokens: int = 4096,
        client: httpx.Client | None = None,
    ):
        load_dotenv(_REPO_ROOT / ".env")
        key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        if not key and client is None:
            raise RuntimeError("NVIDIA_API_KEY is not set (check .env) -- required for the NVIDIA fallback.")
        self.model = model
        self.schema = schema
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.max_tokens = max_tokens
        self._client = client or httpx.Client(
            timeout=timeout, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"}
        )

    def body(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": self.schema.__name__, "schema": self.schema.model_json_schema()},
            },
        }

    def invoke(self, prompt: str) -> BaseModel | None:
        try:
            response = self._client.post(self.url, json=self.body(prompt))
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise HTTPProviderTransient(f"nvidia {self.model}: {type(exc).__name__} (no response)") from exc
        code = response.status_code
        if code == 429 or code >= 500:
            raise HTTPProviderTransient(f"nvidia {self.model}: HTTP {code}", retry_after=_retry_after(response))
        if code >= 400:
            raise HTTPProviderRefused(f"nvidia {self.model}: HTTP {code}: {response.text[:200]}")
        content = (response.json()["choices"][0]["message"].get("content") or "").strip()
        content = _THINK_RE.sub("", content).strip()
        if not content:
            return None
        match = _OBJECT_RE.search(content)
        return self.schema.model_validate_json(match.group(0) if match else content)
