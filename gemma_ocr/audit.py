"""Check that every page/file in an output folder went through the same process.

Groups pages by their pass signature (which passes ran, which thinking variant answered, how each finished),
and flags anything outside the designed path: fallbacks, truncated calls, kept drafts, empty pages,
records from a different pipeline version or config.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def page_signature(r: dict) -> tuple:
    p = r.get("passes") or {}
    sig = ["text-layer" if r["has_text_layer"] else "scan"]
    sig.append("orientation" if r.get("orientation_votes") is not None else "no-orientation")
    for name in ("draft", "proofread", "cross_read"):
        m = p.get(name)
        if not m:
            sig.append(f"{name}:off")
            continue
        variant = "think" if m["think"] else "no-think"
        sig.append(f"{name}:{variant}{'(fallback)' if m.get('fallback') else ''}/{m['finish']}")
    sig.append("proof-used" if r["proofread_used"] else "DRAFT-KEPT")
    if r["has_text_layer"]:
        sig.append("snapped")
    return tuple(sig)


def audit(out: Path) -> dict:
    pages = [json.loads(f.read_text()) for f in sorted((out / "pages").glob("*.json"))]
    docs = [json.loads(f.read_text()) for f in sorted((out / "docs").glob("*.json"))]
    sigs = Counter(page_signature(r) for r in pages)
    issues = []
    for r in pages:
        where = f"{r['file']} p{r['page']}"
        for name, m in (r.get("passes") or {}).items():
            if m.get("fallback"):
                issues.append(f"{where}: {name} answered by fallback ({'think' if m['think'] else 'no-think'})")
            if m.get("finish") != "stop":
                issues.append(f"{where}: {name} finished with {m.get('finish')}")
        if not r["proofread_used"]:
            issues.append(f"{where}: proofread rejected, draft kept")
        if len(r["text"].strip()) < 30:
            issues.append(f"{where}: almost no text ({len(r['text'].strip())} chars), check it is not a photo/blank page")
        if r["strips"] == 0:
            issues.append(f"{where}: no strips (page fits one image: {r['size'][0]}x{r['size'][1]})")
        for w in r["checks"].get("format_warnings", []):
            issues.append(f"{where}: {w}")
    for d in docs:
        if "error" in (d.get("fields") or {}):
            issues.append(f"{d['file']}: field extraction failed")
        elif (d.get("fields_call") or {}).get("fallback"):
            issues.append(f"{d['file']}: fields answered by thinking fallback")
    versions = Counter((r.get("pipeline_version"), r.get("config_fingerprint")) for r in pages + docs)
    field_keys = Counter(tuple(sorted((d.get("fields") or {}).keys())) for d in docs)
    return {"pages": len(pages), "docs": len(docs), "signatures": sigs, "issues": issues,
            "versions": versions, "field_key_sets": len(field_keys)}


def print_audit(out: Path) -> None:
    a = audit(out)
    print(f"{a['pages']} pages, {a['docs']} files")
    print("\npipeline version / config fingerprint:")
    for (v, fp), n in a["versions"].items():
        print(f"  {n:4d}  {v} / {fp}")
    print("\npass signatures (pages):")
    for sig, n in a["signatures"].most_common():
        print(f"  {n:4d}  " + " | ".join(sig))
    print(f"\nfield key sets across files: {a['field_key_sets']} (1 = every file has the same fields)")
    print(f"\n{len(a['issues'])} items to look at:")
    for i in a["issues"]:
        print("  - " + i)
