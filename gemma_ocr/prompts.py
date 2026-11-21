"""Prompt templates. Changing any text here changes the request hash, so cached calls are redone."""
from __future__ import annotations

TRANSCRIBE = """You are an expert OCR transcriber for {doc_context}.

{images_note}

Transcribe ALL text on the page in reading order: top to bottom, and right to left for Persian.
Rules:
- Copy exactly. Do not translate, summarise, fix spelling or complete anything.
- Digits: copy every digit exactly as printed, in the same script (Persian ۰-۹ stay Persian, Latin 0-9 stay Latin). Long numbers (case numbers, national IDs, tracking codes, phone numbers, sheba/IBAN, amounts) must have exactly the printed number of digits. Keep separators (/ , - .) and date order exactly as printed.
- Keep every value on the same line as its label, e.g. «شماره پرونده : ۱۴۰۲...».
- Header areas with side-by-side blocks: transcribe the right block, then the middle, then the left.
- Tables: Markdown table; the first column is the RIGHTMOST printed column; empty cells stay empty.
- Non-text elements, in brackets: [مهر: text inside the stamp if readable], [امضا], [لوگو: ...], [بارکد], [QR].
- Handwriting: transcribe it if legible, otherwise [دست‌نوشته ناخوانا].
- Unreadable text: [ناخوانا]. Never invent text. If one digit is uncertain, still write your best reading and put [?] right after that number.
Output only the transcription in Markdown, with no comments before or after."""

PROOFREAD = """You are proofreading an OCR transcription of a page from {doc_context}.

{images_note}

DRAFT TRANSCRIPTION:
<<<
{draft}
>>>
{layer_note}
Compare the draft with the images line by line and output the corrected, complete transcription.
Check in particular:
- every digit of every number (count the digits of long IDs and amounts), dates, phone numbers;
- that every value sits next to its correct label;
- names and Persian words spelled exactly as printed;
- table rows and columns (first column = rightmost printed column);
- missing lines, and lines duplicated because strips overlap.
Keep the same Markdown format and bracket conventions. Output only the corrected transcription, nothing else."""

LAYER_NOTE = """
The PDF also has an embedded digital text layer for this page (below). Its characters and digits are exact, but
the order of lines and words is often scrambled and some letters can be split or reversed. Use it to verify digits
and spelling; use the images for layout and order. Do not copy its scrambled order.
DIGITAL TEXT LAYER:
<<<
{layer}
>>>
"""

ORIENT = """These {n} images are the same document page rotated in different ways, labelled {labels}.
Which label shows the page upright, so that the text lines are horizontal and Persian/English text can be read normally?
Answer with the single letter only."""

FIELDS = """Below is the OCR transcription of a document ({doc_context}). Extract structured information.
Return one JSON object with exactly these keys (use null or [] when absent; copy numbers and names exactly as in the text, digits in the same script):
{{
 "document_type_fa": "type of document in Persian, e.g. دادنامه / نظریه کارشناسی / ابلاغیه",
 "document_type_en": "same in English",
 "title": "main title/heading as printed",
 "issuer": "court, branch, organisation or company that issued it",
 "case_number": "شماره پرونده",
 "archive_number": "شماره بایگانی",
 "document_number": "شماره دادنامه / شماره نامه / شماره ابلاغیه / invoice or order number",
 "tracking_code": "کد رهگیری / کد پیگیری",
 "dates": [{{"label": "label as printed", "date": "as printed"}}],
 "parties": [{{"role": "e.g. خواهان / خوانده / محکوم له / محکوم علیه / کارشناس / خریدار", "name": "", "father_name": "", "national_id": "", "phone": "", "address": ""}}],
 "subject": "موضوع / خواسته",
 "amounts": [{{"label": "", "amount": "as printed", "currency": "ریال/تومان/..."}}],
 "decision_or_result": "operative part / outcome in Persian, short",
 "summary_en": "3-5 sentence English summary of what this document is and says",
 "summary_fa": "۲ تا ۳ جمله خلاصه فارسی"
}}

TRANSCRIPTION ({name}):
<<<
{text}
>>>
Output only the JSON object."""

FIELD_KEYS = ["document_type_fa", "document_type_en", "title", "issuer", "case_number", "archive_number",
              "document_number", "tracking_code", "dates", "parties", "subject", "amounts", "decision_or_result",
              "summary_en", "summary_fa"]


def images_note(n_strips: int) -> str:
    if not n_strips:
        return "The image is the whole page."
    return (f"Image 1 is the whole page (use it for layout and reading order). Images 2-{n_strips + 1} are "
            f"{n_strips} overlapping full-width horizontal strips of the same page, top to bottom, at higher "
            "resolution: read small text and digits from them. Consecutive strips overlap, so transcribe each line once.")
