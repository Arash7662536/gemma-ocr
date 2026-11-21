"""Gemma (OpenAI-compatible vLLM) client with an on-disk cache keyed by the exact request body.

The cache makes runs resumable and lets a code change be checked for equivalence: if the new code
builds byte-identical requests, every call is a cache hit.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .config import Config


@dataclass
class CallResult:
    text: str
    finish: str | None
    think: bool
    usage: dict | None = None
    secs: float = 0.0
    reasoning_chars: int = 0
    cached: bool = False
    fallback: bool = False            # True when the first variant failed and the other answered

    def meta(self) -> dict:
        return {"finish": self.finish, "think": self.think, "fallback": self.fallback, "cached": self.cached,
                "secs": self.secs, "usage": self.usage}


def clean(text: str) -> str:
    """Strip code fences and the <<< >>> delimiters the model sometimes echoes from the prompt."""
    t = text.strip()
    t = re.sub(r"^```(?:markdown|md)?\s*|\s*```$", "", t)
    t = re.sub(r"^<<<\s*|\s*>>>$", "", t)
    return t.strip()


def looks_bad(res: CallResult, min_chars: int = 20) -> bool:
    """Truncated (reasoning loop / repetition) or empty."""
    t = res.text
    if res.finish == "length" or len(t.strip()) < min_chars:
        return True
    lines = [l for l in t.splitlines() if l.strip()]
    return len(lines) > 30 and len(set(lines)) < 0.3 * len(lines)


@dataclass
class GemmaClient:
    cfg: Config
    session: requests.Session = field(default_factory=requests.Session)

    @property
    def cache(self) -> Path:
        return Path(self.cfg.cache_dir)

    def body(self, content: list, *, think: bool, max_tokens: int, soft_tokens: int | None = None,
             json_mode: bool = False) -> dict:
        body = {
            "model": self.cfg.model,
            "temperature": float(self.cfg.temperature),
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": think},
            "mm_processor_kwargs": {"max_soft_tokens": soft_tokens or self.cfg.soft_tokens},
            "repetition_penalty": self.cfg.repetition_penalty,
            "messages": [{"role": "user", "content": content}],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def call(self, content: list, *, think: bool, max_tokens: int, soft_tokens: int | None = None,
             json_mode: bool = False, tag: str = "") -> CallResult:
        """One chat completion, cached on disk by request hash."""
        body = self.body(content, think=think, max_tokens=max_tokens, soft_tokens=soft_tokens, json_mode=json_mode)
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        path = self.cache / f"{tag}{key[:40]}.json"
        if path.exists():
            d = json.loads(path.read_text())
            return CallResult(d["text"], d["finish"], d["think"], d.get("usage"), d.get("secs", 0.0),
                              d.get("reasoning_chars", 0), cached=True)
        for attempt in range(self.cfg.retries):
            try:
                t = time.time()
                r = self.session.post(self.cfg.endpoint, json=body, timeout=self.cfg.timeout,
                                      headers={"Authorization": f"Bearer {self.cfg.api_key}"})
                r.raise_for_status()
                j = r.json()
                ch = j["choices"][0]
                msg = ch["message"]
                d = {"text": msg.get("content") or "", "finish": ch.get("finish_reason"), "usage": j.get("usage"),
                     "think": think, "secs": round(time.time() - t, 1),
                     "reasoning_chars": len(msg.get("reasoning_content") or msg.get("reasoning") or "")}
                self.cache.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(d, ensure_ascii=False))
                return CallResult(d["text"], d["finish"], think, d["usage"], d["secs"], d["reasoning_chars"])
            except Exception:
                if attempt == self.cfg.retries - 1:
                    raise
                time.sleep(5 * (attempt + 1))
        raise RuntimeError("unreachable")

    def call_robust(self, content: list, *, think_first: bool, max_tokens: int, tag: str,
                    min_chars: int = 20) -> CallResult:
        """Try one thinking mode, fall back to the other if the answer loops, truncates or is empty.

        Thinking reads better but sometimes loops until max_tokens; no-think occasionally repeats itself."""
        res = None
        for i, think in enumerate((think_first, not think_first)):
            res = self.call(content, think=think, max_tokens=max_tokens, tag=tag)
            res.fallback = i == 1
            if not looks_bad(res, min_chars):
                return res
        return res
