"""Static HTML report (+ fields.csv) from an output folder."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from collections import defaultdict
from pathlib import Path

import markdown

from .prompts import FIELD_KEYS
from .verify import norm

NUM_RE = re.compile(r"[\d۰-۹][\d۰-۹,٬/]*[\d۰-۹]")


def group_key(name: str) -> str:
    """Files of the same case share the 14+ digit case number in their name."""
    m = re.search(r"\d{14,}", name)
    return m.group(0) if m else Path(name).stem


def md_html(text: str, flagged: set) -> str:
    h = markdown.markdown(text, extensions=["tables"])

    def mark(m):
        s = m.group(0)
        key = norm(s).replace(",", "").replace("٬", "")
        return f'<mark title="not confirmed by the independent second read">{s}</mark>' if key in flagged else s

    # only touch text between tags
    return re.sub(r">([^<]+)<", lambda m: ">" + NUM_RE.sub(mark, m.group(1)) + "<", h)


def esc(x) -> str:
    return html.escape("" if x is None else str(x))

def fields_html(f):
    if not f or "error" in f:
        return f"<pre>{esc(json.dumps(f, ensure_ascii=False, indent=1))}</pre>"
    rows = []
    for k, v in f.items():
        if isinstance(v, list):
            if not v:
                continue
            if isinstance(v[0], dict):
                cols = [c for c in v[0].keys()]
                cell = "<table class=inner><tr>" + "".join(f"<th>{esc(c)}</th>" for c in cols) + "</tr>" + "".join(
                    "<tr>" + "".join(f"<td>{esc(it.get(c))}</td>" for c in cols) + "</tr>" for it in v if isinstance(it, dict)) + "</table>"
            else:
                cell = "<br>".join(esc(x) for x in v)
        else:
            if v in (None, ""):
                continue
            cell = esc(v)
        rows.append(f"<tr><th>{esc(k)}</th><td dir=auto>{cell}</td></tr>")
    return "<table class=fields>" + "".join(rows) + "</table>"


def check_badge(r):
    c = r["checks"]
    tl = c.get("text_layer")
    parts = []
    if tl:
        parts.append(f"text layer: {tl['layer_numbers_found']}/{tl['numbers_in_layer']} numbers, word recall {tl['word_recall']:.0%}")
    if r["text_layer_fixes"]:
        parts.append(f"{len(r['text_layer_fixes'])} digit fixes from text layer")
    n_unc = len(c["unconfirmed_by_cross_read"])
    parts.append(f"{n_unc} numbers unconfirmed by 2nd read" if n_unc else "all numbers confirmed by 2nd read")
    if c.get("format_warnings"):
        parts.append("⚠ " + "; ".join(c["format_warnings"]))
    if r["rotation_ccw"]:
        parts.append(f"rotated {r['rotation_ccw']}°")
    fb = [f"{k} ({'think' if v.get('think') else 'no-think'})" for k, v in (r.get("passes") or {}).items() if v.get("fallback")]
    if fb:
        parts.append("fallback used: " + ", ".join(fb))
    return " · ".join(parts)


CSS = """
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--card:#fff;--acc:#2f5d8a;--mark:#ffe58a}
@media (prefers-color-scheme:dark){:root{--bg:#18181a;--fg:#e8e6e1;--mut:#9a988f;--line:#333336;--card:#202023;--acc:#8db5e0;--mark:#6b5a12}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,'Vazirmatn','Tahoma',sans-serif;display:flex}
nav{width:300px;height:100vh;overflow:auto;position:sticky;top:0;border-right:1px solid var(--line);padding:12px;flex-shrink:0}
nav input{width:100%;padding:7px 9px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);margin-bottom:8px}
nav .grp{font-size:12px;color:var(--mut);margin:12px 0 2px;font-variant-numeric:tabular-nums}
nav a{display:block;padding:4px 8px;border-radius:5px;color:var(--fg);text-decoration:none;font-size:13px;word-break:break-all}
nav a:hover{background:var(--line)}nav a small{display:block;color:var(--mut);font-size:11px}
main{flex:1;min-width:0;padding:20px 28px}
header h1{margin:0 0 4px;font-size:22px}header p{margin:0 0 18px;color:var(--mut)}
section{border-top:1px solid var(--line);padding-top:16px;margin-bottom:34px}h2{font-size:18px;margin:0 0 6px;word-break:break-all}
.sum{margin:2px 0;color:var(--mut);max-width:1000px}.dup{color:var(--mut);font-size:13px}
details{margin:8px 0 14px}summary{cursor:pointer;color:var(--acc)}
.page{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.15fr);gap:18px;margin:14px 0;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
.img img{width:100%;border:1px solid var(--line);border-radius:4px}
.meta{font-size:12px;color:var(--mut);margin-bottom:6px}
.tabs button{border:1px solid var(--line);background:none;color:var(--fg);padding:3px 10px;border-radius:5px;cursor:pointer;font-size:12px;margin:0 4px 6px 0}
.tabs button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.pane{display:none;max-height:1100px;overflow:auto}.pane.on{display:block}
pre.pane{white-space:pre-wrap;font:13px/1.6 ui-monospace,monospace;background:var(--bg);padding:10px;border-radius:6px}
.t table,.fields,.inner{border-collapse:collapse;margin:6px 0;font-size:13px}.t td,.t th,.fields td,.fields th,.inner td,.inner th{border:1px solid var(--line);padding:3px 7px;vertical-align:top}
.fields th{text-align:left;color:var(--mut);font-weight:500;white-space:nowrap}
mark{background:var(--mark);color:inherit;border-radius:2px}
@media (max-width:900px){body{display:block}nav{width:auto;height:auto;position:static;max-height:40vh}.page{grid-template-columns:1fr}main{padding:12px 16px}}
"""
JS = """
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{const p=b.closest('.txt');
p.querySelectorAll('.tabs button').forEach(x=>x.classList.toggle('on',x===b));
p.querySelectorAll('.pane').forEach(x=>x.classList.toggle('on',x.classList.contains(b.dataset.t)))});
const q=document.getElementById('q');q.oninput=()=>{const v=q.value.trim().toLowerCase();
document.querySelectorAll('nav a').forEach(a=>a.style.display=!v||a.dataset.q.toLowerCase().includes(v)?'':'none')};
"""


def build_report(out: Path, title: str = "OCR results") -> Path:
    docs = {p.stem: json.loads(p.read_text()) for p in sorted((out / "docs").glob("*.json"))}
    pages = defaultdict(list)
    for p in (out / "pages").glob("*.json"):
        r = json.loads(p.read_text())
        pages[r["file"]].append(r)
    for v in pages.values():
        v.sort(key=lambda r: r["page"])

    groups = defaultdict(list)
    for name in docs:
        groups[group_key(name)].append(name)

    nav, body = [], []
    total_pages = sum(len(v) for v in pages.values())
    total_unc = sum(len(r["checks"]["unconfirmed_by_cross_read"]) for v in pages.values() for r in v)
    for g in sorted(groups):
        nav.append(f'<div class=grp>{esc(g)}</div>')
        for name in sorted(groups[g], key=lambda n: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", n)]):
            d = docs[name]
            did = "d" + hashlib.md5(name.encode()).hexdigest()[:10]
            f = d.get("fields") or {}
            unc = sum(len(r["checks"]["unconfirmed_by_cross_read"]) for r in pages[name])
            nav.append(f'<a href="#{did}" data-q="{esc(norm(d["text"])[:20000])} {esc(name)}">{esc(name)}'
                       f'<small>{esc(f.get("document_type_fa") or "")} · {d["pages"]}p{" · ⚠" + str(unc) if unc else ""}</small></a>')
            pg_html = []
            for r in pages[name]:
                flagged = set(r["checks"]["unconfirmed_by_cross_read"])
                img = f"pages/{r['file']}__p{r['page']}.jpg"
                pg_html.append(f"""
    <div class=page>
     <div class=img><a href="{esc(img)}" target=_blank><img loading=lazy src="{esc(img)}"></a></div>
     <div class=txt>
      <div class=meta>page {r['page']} · {esc(check_badge(r))}</div>
      <div class=tabs>
       <button class=on data-t=t>Transcription</button><button data-t=m>Markdown</button><button data-t=dr>Draft</button><button data-t=cr>2nd read</button>{'<button data-t=tl>PDF text layer</button>' if r['has_text_layer'] else ''}
      </div>
      <div class="pane t on" dir=rtl>{md_html(r['text'], flagged)}</div>
      <pre class="pane m" dir=rtl>{esc(r['text'])}</pre>
      <pre class="pane dr" dir=rtl>{esc(r['draft'])}</pre>
      <pre class="pane cr" dir=rtl>{esc(r['cross_read'])}</pre>
      <pre class="pane tl" dir=rtl>{esc(r.get('text_layer', ''))}</pre>
     </div>
    </div>""")
            dup = f"<p class=dup>Identical copies (not processed again): {esc(', '.join(d['duplicates']))}</p>" if d.get("duplicates") else ""
            body.append(f"""
    <section id="{did}">
     <h2>{esc(name)}</h2>
     <p class=sum>{esc(f.get('summary_en') or '')}</p>
     <p class=sum dir=rtl>{esc(f.get('summary_fa') or '')}</p>
     {dup}
     <details><summary>Extracted fields</summary>{fields_html(f)}</details>
     {''.join(pg_html)}
    </section>""")
    doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>{CSS}</style></head><body>
<nav><input id=q placeholder="Search text / file…">{''.join(nav)}</nav>
<main><header><h1>{esc(title)}</h1>
<p>{len(docs)} unique files · {total_pages} pages · local Gemma-4 31B (vLLM, 1120 vision tokens/image) ·
<mark>highlighted</mark> numbers were not confirmed by an independent second read ({total_unc} in total), so check them against the image.</p></header>
{''.join(body)}</main><script>{JS}</script></body></html>"""
    (out / "report.html").write_text(doc)

    with open(out / "fields.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        lists = ["dates", "parties", "amounts"]
        cols = [k for k in FIELD_KEYS if k not in lists]
        w.writerow(["file", "pages", "case_group"] + cols + lists)
        for name, d in sorted(docs.items()):
            f = d.get("fields") or {}
            w.writerow([name, d["pages"], group_key(name)] + [f.get(c) or "" for c in cols]
                       + [json.dumps(f.get(k) or [], ensure_ascii=False) for k in lists])
    return out / "report.html"
