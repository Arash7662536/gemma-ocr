"""gemma-ocr command line.

  gemma-ocr run INPUT_DIR OUT_DIR [--config c.toml] [--set key=value ...] [--only a.pdf,b.jpg] [--no-report]
  gemma-ocr report OUT_DIR [--title "..."]
  gemma-ocr audit OUT_DIR
  gemma-ocr serve OUT_DIR [--port 8799] [--user aiteam]   (password from --password or $OCR_REPORT_PASSWORD)
  gemma-ocr eval INPUT_DIR FILE [FILE ...]
  gemma-ocr config [--config c.toml] [--set ...]           (print the effective config)
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gemma-ocr", description="Persian document OCR with a local Gemma-4 (vLLM).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def with_config(p):
        p.add_argument("--config", help="TOML file with Config fields")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override one setting")

    p = sub.add_parser("run", help="OCR every PDF/JPG/PNG in a folder")
    p.add_argument("input_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--only", default="", help="comma-separated file names")
    p.add_argument("--no-report", action="store_true")
    with_config(p)

    p = sub.add_parser("report", help="(re)build report.html and fields.csv")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--title", default=None)
    with_config(p)

    p = sub.add_parser("audit", help="check every page went through the same passes")
    p.add_argument("out_dir", type=Path)

    p = sub.add_parser("serve", help="serve the report with basic auth on 127.0.0.1")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--port", type=int, default=8799)
    p.add_argument("--user", default="aiteam")
    p.add_argument("--password", default=os.environ.get("OCR_REPORT_PASSWORD"))

    p = sub.add_parser("eval", help="A/B reading setups against PDF text layers")
    p.add_argument("input_dir", type=Path)
    p.add_argument("files", nargs="+")
    p.add_argument("--out", type=Path, default=None)
    with_config(p)

    p = sub.add_parser("config", help="print the effective config")
    with_config(p)

    a = ap.parse_args(argv)

    if a.cmd == "run":
        from .pipeline import run
        from .report import build_report
        cfg = load_config(a.config, a.set)
        if not cfg.api_key:
            print("warning: GEMMA_KEY is not set", file=sys.stderr)
        m = run(cfg, a.input_dir, a.out_dir, [x for x in a.only.split(",") if x] or None)
        if not a.no_report:
            print("report:", build_report(a.out_dir, cfg.report_title))
        return 1 if m["failures"] else 0
    if a.cmd == "report":
        from .report import build_report
        cfg = load_config(a.config, a.set)
        print("report:", build_report(a.out_dir, a.title or cfg.report_title))
    elif a.cmd == "audit":
        from .audit import print_audit
        print_audit(a.out_dir)
    elif a.cmd == "serve":
        from .serve import serve
        pw = a.password or secrets.token_urlsafe(9)
        if not a.password:
            print(f"generated password: {pw}", flush=True)
        serve(str(a.out_dir), a.port, a.user, pw)
    elif a.cmd == "eval":
        from .evaluate import evaluate
        evaluate(load_config(a.config, a.set), a.input_dir, a.files, a.out)
    elif a.cmd == "config":
        print(json.dumps(load_config(a.config, a.set).public(), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
