"""All pipeline settings in one place. Defaults are the settings used for the first production run (2026-11-18).

Load order: defaults -> TOML file (`--config`) -> environment (GEMMA_URL, GEMMA_KEY, GEMMA_MODEL) -> `--set key=value`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DOC_CONTEXT = ("Persian (Farsi) documents from Iran: court and judiciary papers (دادنامه، اجرای احکام، اجراییه، "
               "ابلاغیه، نظریه کارشناسی، اظهارنامه، لایحه، صورتجلسه، روند کار پرونده)، official gazette / company "
               "registration notices (روزنامه رسمی)، invoices and online-shop order receipts")


@dataclass
class Config:
    # --- server
    endpoint: str = "http://127.0.0.1:8001/v1/chat/completions"
    model: str = "google/gemma-4-31B-it"
    api_key: str = field(default="", repr=False)
    timeout: int = 1800
    retries: int = 4

    # --- vision budget. Gemma-4: 16px patches, 3x3 pooling -> 1 soft token = 48x48 px.
    # 1120 soft tokens (the max) ~= 2.58 MP seen at full detail. Needs vLLM --trust-request-mm-kwargs.
    soft_tokens: int = 1120
    orient_soft_tokens: int = 280

    # --- rendering / image prep
    dpi: int = 300
    crop_threshold: int = 200          # grey level below which a pixel counts as ink
    crop_pad: int = 24
    strip_overlap: float = 0.18
    max_strips: int = 6
    min_strips: int = 2                # every page gets overview + at least this many strips (0 = only when needed)
    upscale_below_width: int = 1700    # pages narrower than this are upscaled before cutting strips
    upscale_width: int = 2600
    upscale_max: float = 3.0
    jpeg_quality: int = 92
    orient_px: int = 900_000
    orient_jpeg_quality: int = 85
    preview_px: int = 1_800_000

    # --- sampling
    temperature: float = 0.0
    repetition_penalty: float = 1.03

    # --- per-pass token limits
    max_tokens_draft: int = 9000
    max_tokens_proof: int = 12000
    max_tokens_cross: int = 8000
    max_tokens_fields: int = 4000
    max_tokens_orient: int = 5

    # --- which passes run
    orientation: bool = True           # only for pages without a text layer (born-digital pages are upright)
    proofread: bool = True
    cross_read: bool = True
    snap_to_text_layer: bool = True
    fields: bool = True

    # --- thresholds
    min_text_layer_chars: int = 30     # a PDF page with more than this counts as having a text layer
    proof_min_ratio: float = 0.6       # proofread shorter than this x draft -> keep the draft
    layer_max_chars: int = 12000
    fields_max_chars: int = 40000

    # --- run
    workers: int = 16
    cache_dir: str = "cache"
    doc_context: str = DOC_CONTEXT
    report_title: str = "OCR results"

    # ------------------------------------------------------------------
    @property
    def px_per_image(self) -> int:
        return self.soft_tokens * 48 * 48

    def fingerprint(self) -> str:
        """Hash of every setting that changes model inputs (not workers/paths/key), stored in each page record."""
        skip = {"api_key", "workers", "cache_dir", "report_title", "endpoint", "timeout", "retries"}
        d = {k: v for k, v in dataclasses.asdict(self).items() if k not in skip}
        return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]

    def public(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("api_key")
        return d

    def set(self, key: str, raw: str) -> None:
        if key not in {f.name for f in dataclasses.fields(self)}:
            raise KeyError(f"unknown config key: {key}")
        cur = getattr(self, key)
        if isinstance(cur, bool):
            val = raw.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            val = int(raw)
        elif isinstance(cur, float):
            val = float(raw)
        else:
            val = raw
        setattr(self, key, val)


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env reader: KEY=VALUE or `export KEY=VALUE` lines; existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def load_config(path: str | None = None, overrides: list[str] | None = None) -> Config:
    cfg = Config()
    if path:
        data = tomllib.loads(Path(path).read_text())
        for k, v in data.items():
            cfg.set(k, str(v))
    load_dotenv()
    env = {"GEMMA_URL": "endpoint", "GEMMA_KEY": "api_key", "GEMMA_MODEL": "model"}
    for var, key in env.items():
        if os.environ.get(var):
            setattr(cfg, key, os.environ[var])
    for item in overrides or []:
        k, _, v = item.partition("=")
        cfg.set(k.strip(), v.strip())
    return cfg
