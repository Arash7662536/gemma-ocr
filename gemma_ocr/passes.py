"""The model passes. Each takes prepared images/text and returns its output plus call metadata.

Per page:  orientation (scans only) -> draft -> proofread -> cross-read
Per file:  fields
"""
from __future__ import annotations

import json
import re

from PIL import Image

from . import prompts
from .client import CallResult, GemmaClient, clean, looks_bad
from .config import Config
from .render import fit, image_parts, rotate


def orientation(client: GemmaClient, img: Image.Image) -> tuple[int, list[int]]:
    """Gemma picks the upright one of 4 rotations, twice with different label orders.
    Rotate only if both votes agree. Returns (degrees CCW, votes)."""
    cfg = client.cfg
    small = fit(img, cfg.orient_px)
    labels = "ABCD"
    votes = []
    for order in ([0, 90, 180, 270], [180, 270, 0, 90]):
        content = []
        for k, deg in enumerate(order):
            content += [{"type": "text", "text": f"Image {labels[k]}:"}] + image_parts([rotate(small, deg)], cfg.orient_jpeg_quality)
        content.append({"type": "text", "text": prompts.ORIENT.format(n=4, labels="A, B, C, D")})
        r = client.call(content, think=False, max_tokens=cfg.max_tokens_orient, soft_tokens=cfg.orient_soft_tokens,
                        tag="orient_")
        m = re.search(r"[ABCD]", r.text.upper())
        votes.append(order[labels.index(m.group(0))] if m else 0)
    return (votes[0] if votes[0] == votes[1] else 0), votes


def draft(client: GemmaClient, parts: list, n_strips: int) -> CallResult:
    """Main transcription from overview + strips. Thinking first, no-think fallback."""
    cfg = client.cfg
    text = prompts.TRANSCRIBE.format(doc_context=cfg.doc_context, images_note=prompts.images_note(n_strips))
    return client.call_robust(parts + [{"type": "text", "text": text}], think_first=True,
                              max_tokens=cfg.max_tokens_draft, tag="tr_")


def proofread(client: GemmaClient, parts: list, n_strips: int, draft_text: str, layer: str) -> CallResult:
    """Same images + the raw draft (+ the PDF text layer if any). Thinking first, no-think fallback."""
    cfg = client.cfg
    layer_note = prompts.LAYER_NOTE.format(layer=layer.strip()[:cfg.layer_max_chars]) if layer else ""
    text = prompts.PROOFREAD.format(doc_context=cfg.doc_context, images_note=prompts.images_note(n_strips),
                                    draft=draft_text, layer_note=layer_note)
    return client.call_robust(parts + [{"type": "text", "text": text}], think_first=True,
                              max_tokens=cfg.max_tokens_proof, tag="pr_")


def choose_final(draft_res: CallResult, proof_res: CallResult | None, cfg: Config) -> tuple[str, str, bool]:
    """(final, draft_text, proofread_used). Keep the draft if the proofread looks broken or much shorter."""
    draft_text = clean(draft_res.text)
    if proof_res is None:
        return draft_text, draft_text, False
    final = clean(proof_res.text)
    if looks_bad(proof_res) or len(final) < cfg.proof_min_ratio * len(draft_text):
        return draft_text, draft_text, False
    return final, draft_text, True


def cross_read(client: GemmaClient, overview_parts: list) -> CallResult:
    """Independent second read: one image, no strips, no thinking -> different errors from the main read.
    Used only to flag numbers it does not confirm."""
    cfg = client.cfg
    text = prompts.TRANSCRIBE.format(doc_context=cfg.doc_context, images_note=prompts.images_note(0))
    res = client.call_robust(overview_parts + [{"type": "text", "text": text}], think_first=False,
                             max_tokens=cfg.max_tokens_cross, tag="cr_")
    res.text = clean(res.text)
    return res


def fields(client: GemmaClient, name: str, text: str) -> tuple[dict, dict]:
    """Structured fields for one file from its final transcription (text only, JSON mode).
    No-think first, thinking fallback if the JSON does not parse. Returns (fields, call meta)."""
    cfg = client.cfg
    prompt = prompts.FIELDS.format(doc_context=cfg.doc_context, name=name, text=text[:cfg.fields_max_chars])
    r = None
    for i, think in enumerate((False, True)):
        r = client.call([{"type": "text", "text": prompt}], think=think, max_tokens=cfg.max_tokens_fields,
                        json_mode=True, tag="fx_")
        r.fallback = i == 1
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", r.text.strip())
        try:
            return json.loads(t), r.meta()
        except Exception:
            continue
    return {"error": "could not parse JSON", "raw": r.text[:2000]}, r.meta()
