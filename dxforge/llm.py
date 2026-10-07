"""Optional, pluggable LLM provider.

DxForge runs fully WITHOUT an LLM. The provider is an optional upgrade that only
improves natural-language intent parsing. Controlled by env vars:

    LLM_PROVIDER = none | ollama | openai      (default: none)
    OLLAMA_HOST  = http://localhost:11434       (ollama)
    OLLAMA_MODEL = llama3.1                      (ollama)
    OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL  (openai-compatible)

If a provider is selected but unreachable, callers fall back to the deterministic
parser — the app never hard-fails because a model is missing.
"""
from __future__ import annotations

import json
import os
from typing import Optional


def provider_name() -> str:
    return os.getenv("LLM_PROVIDER", "none").lower()


def is_enabled() -> bool:
    return provider_name() in ("ollama", "openai")


def complete_json(system: str, user: str, timeout: float = 20.0) -> Optional[dict]:
    """Ask the configured provider for a JSON object. Returns None on any failure."""
    prov = provider_name()
    try:
        if prov == "ollama":
            return _ollama(system, user, timeout)
        if prov == "openai":
            return _openai(system, user, timeout)
    except Exception:
        return None
    return None


def _extract_json(text: str) -> Optional[dict]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def _ollama(system: str, user: str, timeout: float) -> Optional[dict]:
    import requests
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    r = requests.post(
        f"{host}/api/chat",
        json={"model": model, "stream": False, "format": "json",
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}]},
        timeout=timeout,
    )
    r.raise_for_status()
    return _extract_json(r.json()["message"]["content"])


def _openai(system: str, user: str, timeout: float) -> Optional[dict]:
    import requests
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    r = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "temperature": 0,
              "response_format": {"type": "json_object"},
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}]},
        timeout=timeout,
    )
    r.raise_for_status()
    return _extract_json(r.json()["choices"][0]["message"]["content"])
