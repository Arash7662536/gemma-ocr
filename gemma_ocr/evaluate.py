"""A/B the reading setups on born-digital PDF pages, whose embedded text layer is exact ground truth.

Measures recall/precision of the layer's numbers (>= 4 digits, dates) and of its words in each transcription.
The proofread variant gets no text layer, so it simulates what scanned pages get.
"""
from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import json
from collections import Counter
from pathlib import Path

from . import prompts
from .client import GemmaClient, clean
from .config import Config
from .render import crop_margins, image_parts, load_pages, page_images
from .verify import norm, numbers, words

SETUPS = {  # name: (use strips, thinking, soft tokens per image)
    "single_280": (False, False, 280),
    "single_1120": (False, False, 1120),
    "strips_1120": (True, False, 1120),
    "strips_1120_think": (True, True, 1120),
}


def score(text: str, layer: str) -> dict:
    ln, on = set(numbers(layer)), set(numbers(text))
    fw, lw = Counter(words(text)), Counter(words(layer))
    c = sum((fw & lw).values())
    return {"num_recall": len(ln & on) / max(1, len(ln)), "num_prec": len(ln & on) / max(1, len(on)),
            "word_recall": c / max(1, sum(lw.values())), "word_prec": c / max(1, sum(fw.values()))}


def evaluate(cfg: Config, input_dir: Path, names: list[str], out_json: Path | None = None) -> list[dict]:
    pages = []
    for n in names:
        for pg in load_pages(input_dir / n, cfg.dpi):
            if len(norm(pg["text_layer"]).strip()) > 200:
                pages.append((n, pg["page"], crop_margins(pg["img"], cfg.crop_threshold, cfg.crop_pad), pg["text_layer"]))
    print(len(pages), "eval pages")

    def job(p):
        n, pno, img, layer = p
        row = {"page": f"{n} p{pno}"}
        for k, (use_strips, think, soft) in SETUPS.items():
            c = GemmaClient(dataclasses.replace(cfg, soft_tokens=soft))
            imgs, ns = page_images(img, c.cfg, use_strips)
            parts = image_parts(imgs, cfg.jpeg_quality)
            note = prompts.images_note(ns)
            r = c.call(parts + [{"type": "text", "text": prompts.TRANSCRIBE.format(doc_context=cfg.doc_context, images_note=note)}],
                       think=think, max_tokens=cfg.max_tokens_draft, tag="ab_")
            row[k] = score(clean(r.text), layer)
            if k == "strips_1120_think":  # + proofread without the text layer
                pr = c.call(parts + [{"type": "text", "text": prompts.PROOFREAD.format(
                    doc_context=cfg.doc_context, images_note=note, draft=r.text, layer_note="")}],
                    think=True, max_tokens=cfg.max_tokens_proof, tag="ab_")
                row["think+proof"] = score(clean(pr.text), layer)
        print(json.dumps({"page": row["page"], **{k: round(v["num_recall"], 2) for k, v in row.items() if k != "page"}}), flush=True)
        return row

    with cf.ThreadPoolExecutor(cfg.workers) as ex:
        rows = list(ex.map(job, pages))
    if out_json:
        out_json.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print("\nsetup                num_recall num_prec word_recall word_prec")
    for k in list(SETUPS) + ["think+proof"]:
        m = {x: sum(r[k][x] for r in rows) / len(rows) for x in ("num_recall", "num_prec", "word_recall", "word_prec")}
        print(f"{k:20s} {m['num_recall']:.3f}      {m['num_prec']:.3f}    {m['word_recall']:.3f}       {m['word_prec']:.3f}")
    return rows
