# gemma-ocr

OCR for Persian documents (Iranian court and judiciary papers, gazette notices, invoices, online-shop receipts)
with a **local** Gemma-4 31B served by vLLM. No cloud model sees the documents.

Each page is read several times by the model and checked by code; the output is a Markdown transcription per page,
structured fields per file, and a static HTML report that marks every number that was not confirmed.

## Pipeline

```
file ─► render ─► orientation ─► crop ─► overview + strips ─► draft ─► proofread ─► snap ─► cross-read ─► checks
                   (scans only)                                 (think)   (think)    (PDF text   (no-think)
                                                                                     layer only)
all pages of a file ─► fields (JSON)            output folder ─► report.html + fields.csv
```

| step | what | model calls |
|---|---|---|
| render | PDF pages at 300 dpi (`pymupdf`), images as-is (EXIF-rotated); the PDF's own text layer is kept | – |
| orientation | page shown in 4 rotations labelled A–D, Gemma picks the upright one; asked twice with different label orders, rotated only if both agree. Skipped for pages with a text layer (always upright). | 2 × (4 images @ 280 tokens) |
| crop | white/scanner border trimmed | – |
| overview + strips | the page downscaled to one image, plus 2–6 overlapping (18%) **full-width** horizontal strips, each at native resolution (`min_strips = 2`: every page gets at least 2). Pages narrower than 1700 px are upscaled to 2600 px first. Full width keeps label/value pairs and table rows intact. | – |
| draft | full transcription from overview + strips, thinking on; falls back to thinking off if the answer loops or truncates | 1 (or 2) |
| proofread | same images + the draft (+ the PDF text layer when there is one, as a reference for exact characters); thinking on, same fallback. If the result looks broken or is < 60% of the draft, the draft is kept. | 1 (or 2) |
| snap | text-layer pages only: numbers 1–2 digits off (dates: 1 digit) from a text-layer value are replaced by it, only when the nearest value is unique | – |
| cross-read | independent read from the overview alone, thinking off (falls back to on). Not used as text; numbers in the final text it does not contain are **flagged** | 1 (or 2) |
| checks | flagged numbers, `[?]` marks, Iranian mobile (11 digits) / national ID (10 digits) lengths, text-layer recall | – |
| fields | per file, from the final text: type, issuer, case / archive / document numbers, tracking code, dates, parties, subject, amounts, outcome, English + Persian summary (JSON mode; thinking fallback if the JSON does not parse) | 1 (or 2) |

Every call is cached on disk by the hash of its exact request body, so runs are resumable, and a refactor that
builds the same requests reruns entirely from cache. Each page record stores which variant of each pass answered,
`pipeline_version` and a `config_fingerprint`; `gemma-ocr audit` groups pages by that and lists every deviation.

## Why these settings (measured)

The vision budget matters most. Gemma-4 accepts 70/140/280/560/1120 soft tokens per image (1 token = 48×48 px).
Hosted APIs fix it at ~280, which is too coarse for small Persian digits. Locally we use the maximum, 1120.

A/B on 15 born-digital PDF pages, scored against their text layer (exact ground truth), `gemma-ocr eval`:

| setup | numbers found | words found |
|---|---|---|
| 1 image, 280 tokens (hosted-API equivalent) | 46% | 63% |
| 1 image, 1120 tokens | 90% | 80% |
| overview + strips, 1120 tokens | 92% | 80% |
| strips + thinking + proofread (no text layer) | **94%** | 80% |

Word scores are capped by two PDFs whose text layer is itself garbled. In production, text-layer pages also get
the layer in the proofread and the snap step: on the first 79-file run, 402 of 417 text-layer numbers matched exactly
(most of the rest are dates the PDF stores in pieces).

## Setup

GPU server with vLLM ≥ 0.32 (RTX PRO 6000 96 GB: Gemma-4 31B fp8 with 16 parallel sequences).

```bash
cp .env.example .env              # set VLLM_API_KEY / GEMMA_KEY (same value) and OCR_REPORT_PASSWORD
tmux new -s gemma scripts/start_vllm.sh
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

`start_vllm.sh` sets `--mm-processor-kwargs '{"max_soft_tokens": 1120}' --trust-request-mm-kwargs`; without the
second flag vLLM rejects the per-request token budget the client sends. Stop vLLM by killing its `vllm serve`
process (killing the tmux session alone leaves it running behind `tee`).

## Usage

```bash
gemma-ocr run input/ out/                    # OCR every .pdf/.jpg/.jpeg/.png, then build out/report.html
gemma-ocr run input/ out/ --only a.pdf,b.jpg --set workers=8 --config my.toml
gemma-ocr audit out/                         # did every page go through the same passes?
gemma-ocr report out/ --title "Case files"   # rebuild report.html + fields.csv
gemma-ocr serve out/ --port 8799             # basic auth, 127.0.0.1 only
gemma-ocr eval input/ a.pdf b.pdf            # A/B the reading setups on text-layer pages
gemma-ocr config --set dpi=200               # print the effective config
```

Share the report with a Cloudflare quick tunnel (no account needed; the URL dies with the process):

```bash
cloudflared tunnel --config /dev/null --no-autoupdate --url http://127.0.0.1:8799
```

`--config /dev/null` matters on machines with a named tunnel in `~/.cloudflared/config.yml`; without it the quick
tunnel joins that named tunnel instead.

Settings: every field of `gemma_ocr/config.py` can be set in a TOML file (`config.example.toml`) or with
`--set key=value`. `GEMMA_URL`, `GEMMA_KEY`, `GEMMA_MODEL` come from the environment or `.env`.

## Output

```
out/
  report.html, fields.csv, run.json (config, counts, failures), duplicates.json
  pages/<file>__p<n>.jpg     preview image
  pages/<file>__p<n>.json    text, draft, cross_read, text_layer, checks, passes (variant/finish per call), ...
  docs/<file>.json / .md     all pages joined + extracted fields
cache/                       one JSON per model call
```

Identical input files (same md5) are processed once and listed as duplicates; macOS `._*` files are skipped.

## Limits

- Scanned pages have no ground truth: numbers the cross-read did not confirm are highlighted, but a number both
  reads got wrong the same way is not caught. Long IDs and phone numbers have no arithmetic to check them against.
- The orientation vote only rotates when both votes agree; a sideways page with disagreeing votes stays sideways
  (shown in `orientation_votes`).
- Photos with almost no text (e.g. a parcel) are transcribed as the few visible words; `audit` lists them.
- The documents are sensitive: `.gitignore` excludes data, outputs, the cache and `.env`. Keep it that way.

## Development

```bash
.venv/bin/pytest            # no GPU or network: the pipeline test runs against a fake Gemma
```

Layout: `config.py` settings · `render.py` images · `client.py` model calls + cache · `prompts.py` ·
`passes.py` one function per model pass · `verify.py` deterministic checks · `pipeline.py` orchestration ·
`report.py` · `audit.py` · `evaluate.py` · `serve.py` · `cli.py`.
