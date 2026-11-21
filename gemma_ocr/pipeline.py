"""Orchestration: files -> pages -> passes -> page/doc records on disk.

Every page goes through the same steps; the only designed branch is whether the page has a PDF text layer:

  step                 scanned page / image            born-digital PDF page (text layer)
  render               300 dpi / as-is                 300 dpi
  orientation          Gemma vote (2 calls, 280 tok)   skipped (upright by construction)
  crop + strips        yes                             yes
  draft                think -> no-think fallback      same
  proofread            think -> no-think fallback      same, prompt also gets the text layer
  snap to text layer   -                               numbers/dates 1-2 digits off are replaced
  cross-read           no-think -> think fallback      same
  checks               cross-read, format              + text-layer recall

Records keep which variant of each pass answered, so `gemma-ocr audit` can show any page that differs.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import time
import traceback
from pathlib import Path

from . import __version__, passes
from .client import GemmaClient
from .config import Config
from .render import SUPPORTED, crop_margins, fit, image_parts, load_pages, page_images, rotate
from .verify import checks, norm, numbers, snap_to_layer


def list_inputs(input_dir: Path, only: list[str] | None = None) -> tuple[list[Path], dict[str, str]]:
    """Supported files (macOS ._ files skipped), identical files deduplicated by md5 -> (unique, dup_of)."""
    files = sorted(p for p in input_dir.iterdir() if p.is_file() and not p.name.startswith("._")
                   and p.suffix.lower() in SUPPORTED)
    if only:
        files = [f for f in files if f.name in only]
    seen, dup_of, uniq = {}, {}, []
    for f in files:
        h = hashlib.md5(f.read_bytes()).hexdigest()
        if h in seen:
            dup_of[f.name] = seen[h]
        else:
            seen[h] = f.name
            uniq.append(f)
    return uniq, dup_of


def process_page(client: GemmaClient, name: str, pg: dict, out_dir: Path) -> dict:
    cfg = client.cfg
    img, layer = pg["img"], pg["text_layer"]
    has_layer = len(norm(layer).strip()) > cfg.min_text_layer_chars
    meta: dict = {}

    rot, votes = 0, None
    if cfg.orientation and not has_layer:
        rot, votes = passes.orientation(client, img)
    img = crop_margins(rotate(img, rot), cfg.crop_threshold, cfg.crop_pad)
    stem = f"{name}__p{pg['page']}"
    fit(img, cfg.preview_px).save(out_dir / "pages" / f"{stem}.jpg", quality=85)

    imgs, n_strips = page_images(img, cfg)
    parts = image_parts(imgs, cfg.jpeg_quality)

    d = passes.draft(client, parts, n_strips)
    meta["draft"] = d.meta()
    p = None
    if cfg.proofread:
        p = passes.proofread(client, parts, n_strips, d.text, layer if has_layer else "")
        meta["proofread"] = p.meta()
    final, draft_text, used_proof = passes.choose_final(d, p, cfg)

    fixes = []
    if has_layer and cfg.snap_to_text_layer:
        final, fixes = snap_to_layer(final, set(numbers(layer)))

    cross_text = None
    if cfg.cross_read:
        c = passes.cross_read(client, parts[:1])
        meta["cross_read"] = c.meta()
        cross_text = c.text

    rec = {
        "file": name, "page": pg["page"], "rotation_ccw": rot, "orientation_votes": votes,
        "size": list(img.size), "strips": n_strips, "has_text_layer": has_layer,
        "text": final, "draft": draft_text, "cross_read": cross_text or "",
        "proofread_used": used_proof, "text_layer_fixes": fixes,
        "checks": checks(final, cross_text, layer, cfg.min_text_layer_chars),
        "passes": meta,
        "text_layer": layer if has_layer else "",
        "pipeline_version": __version__, "config_fingerprint": cfg.fingerprint(),
    }
    (out_dir / "pages" / f"{stem}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    fb = [k for k, v in meta.items() if v.get("fallback")]
    print(f"  {stem}: rot={rot} strips={n_strips} chars={len(final)} proof={used_proof} fixes={len(fixes)} "
          f"unconfirmed={len(rec['checks']['unconfirmed_by_cross_read'])}{' fallback=' + ','.join(fb) if fb else ''}",
          flush=True)
    return rec


def process_file(client: GemmaClient, f: Path, page_recs: list[dict], dups: list[str], out_dir: Path) -> dict:
    text = "\n\n".join(f"<!-- page {r['page']} -->\n{r['text']}" for r in page_recs)
    flds, call = passes.fields(client, f.name, text) if client.cfg.fields else ({}, None)
    doc = {"file": f.name, "pages": len(page_recs), "duplicates": dups, "fields": flds, "text": text,
           "fields_call": call, "pipeline_version": __version__, "config_fingerprint": client.cfg.fingerprint()}
    (out_dir / "docs" / f"{f.name}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    (out_dir / "docs" / f"{f.name}.md").write_text(f"# {f.name}\n\n{text}\n")
    print(f"  fields: {f.name}", flush=True)
    return doc


def run(cfg: Config, input_dir: Path, out_dir: Path, only: list[str] | None = None) -> dict:
    t0 = time.time()
    client = GemmaClient(cfg)
    (out_dir / "pages").mkdir(parents=True, exist_ok=True)
    (out_dir / "docs").mkdir(exist_ok=True)

    uniq, dup_of = list_inputs(input_dir, only)
    print(f"{len(uniq) + len(dup_of)} files, {len(uniq)} unique", flush=True)
    jobs = [(f.name, pg) for f in uniq for pg in load_pages(f, cfg.dpi)]
    print(f"{len(jobs)} pages", flush=True)

    results, failures = {}, []
    with cf.ThreadPoolExecutor(cfg.workers) as ex:
        futs = {ex.submit(process_page, client, n, pg, out_dir): (n, pg["page"]) for n, pg in jobs}
        for fu in cf.as_completed(futs):
            n, p = futs[fu]
            try:
                results[(n, p)] = fu.result()
            except Exception:
                failures.append({"file": n, "page": p, "error": traceback.format_exc()})
                print(f"FAILED {n} p{p}\n{failures[-1]['error']}", flush=True)

    def do_file(f: Path):
        recs = sorted((r for (n, _), r in results.items() if n == f.name), key=lambda r: r["page"])
        if recs:
            process_file(client, f, recs, [d for d, o in dup_of.items() if o == f.name], out_dir)

    with cf.ThreadPoolExecutor(cfg.workers) as ex:
        list(ex.map(do_file, uniq))

    (out_dir / "duplicates.json").write_text(json.dumps(dup_of, ensure_ascii=False, indent=1))
    manifest = {"pipeline_version": __version__, "config_fingerprint": cfg.fingerprint(), "config": cfg.public(),
                "input_dir": str(input_dir), "files": len(uniq) + len(dup_of), "unique_files": len(uniq),
                "pages": len(jobs), "pages_ok": len(results), "failures": failures,
                "seconds": round(time.time() - t0, 1)}
    (out_dir / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print("ALLDONE", flush=True)
    return manifest
