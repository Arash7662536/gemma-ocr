"""Deterministic checks (no model calls): normalisation, number extraction, text-layer snapping, sanity checks."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

FA_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
AR_FIX = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه", "أ": "ا", "إ": "ا", "آ": "ا", "ؤ": "و"})
BIDI = re.compile("[‎‏‪-‮⁦-⁩‌‍﻿]")
D = r"[\d۰-۹٠-٩]"                                  # any digit script
NUM = r"[\d۰-۹٠-٩][\d۰-۹٠-٩,٬]*[\d۰-۹٠-٩]"         # number with optional thousands separators
DATE = rf"{D}{{2,4}}/{D}{{1,2}}/{D}{{1,4}}"


def norm(text: str) -> str:
    """For comparison only: NFKC (undo presentation forms), Latin digits, unify Arabic/Persian letters."""
    text = unicodedata.normalize("NFKC", text).translate(FA_DIGITS).translate(AR_FIX)
    text = BIDI.sub("", text)
    return re.sub(r"[ً-ٰٟ]", "", text)  # harakat


def numbers(text: str) -> list[str]:
    """Numbers with >= 4 digits (thousands separators dropped) and dates as 'y/m/d', normalised to Latin digits."""
    t = norm(text)
    out = []
    for m in re.finditer(r"\d{2,4}/\d{1,2}/\d{1,4}|\d[\d,٬]*\d", t):
        s = m.group(0)
        if "/" in s:
            out.append(s)
            continue
        d = re.sub(r"[,٬]", "", s)
        if len(d) >= 4:
            out.append(d)
    return out


def words(text: str) -> list[str]:
    t = norm(text)
    t = re.sub(r"<[^<>\n]+>", " ", t)
    return re.findall(r"[؀-ۿ]{2,}|[A-Za-z]{2,}|\d+", t)


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def to_script(digits: str, like: str) -> str:
    """Write Latin digits in the digit script used by `like`."""
    if re.search("[۰-۹]", like):
        return digits.translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))
    return digits


def snap_to_layer(text: str, layer_nums: set) -> tuple[str, list]:
    """Replace OCR numbers/dates missing from the exact PDF text layer by the closest layer value.

    Numbers: <= 1 digit edit (< 7 digits) or <= 2 (longer), and the closest candidate must be unique.
    Dates: exactly 1 edit, same length, unique. Ambiguous or far readings are left alone."""
    fixes = []

    def repl(m):
        raw = m.group(0)
        n = numbers(raw)
        if len(n) != 1 or "/" in n[0] or n[0] in layer_nums:
            return raw
        d = n[0]
        cands = sorted((edit_distance(d, x), x) for x in layer_nums if "/" not in x and abs(len(x) - len(d)) <= 1)
        if not cands or cands[0][0] > (1 if len(d) < 7 else 2) or (len(cands) > 1 and cands[1][0] == cands[0][0]):
            return raw
        best = cands[0][1]
        best_fmt = f"{int(best):,}" if ("," in norm(raw) or "٬" in raw) else best
        fixes.append({"ocr": d, "text_layer": best})
        return to_script(best_fmt, raw)

    layer_dates = {x for x in layer_nums if "/" in x}

    def repl_date(m):
        raw = m.group(0)
        d = norm(raw)
        if d in layer_dates:
            return raw
        cands = sorted((edit_distance(d, x), x) for x in layer_dates if len(x) == len(d))
        if not cands or cands[0][0] != 1 or (len(cands) > 1 and cands[1][0] == 1):
            return raw
        fixes.append({"ocr": d, "text_layer": cands[0][1]})
        return to_script(cands[0][1], raw)

    text = re.sub(rf"(?<![\d۰-۹٠-٩/]){DATE}(?![\d۰-۹٠-٩/])", repl_date, text)
    out, pos = [], 0  # numbers outside dates
    for m in re.finditer(DATE, text):
        out.append(re.sub(NUM, repl, text[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(re.sub(NUM, repl, text[pos:]))
    return "".join(out), fixes


def format_warnings(text: str) -> list[str]:
    """Iranian mobiles are 11 digits (09...), national IDs (کد ملی) 10 digits."""
    warn = []
    t = norm(text)
    for m in re.finditer(r"(?<!\d)09\d{7,11}(?!\d)", t):
        if len(m.group(0)) != 11:
            warn.append(f"mobile with {len(m.group(0))} digits: {m.group(0)}")
    for m in re.finditer(r"کد\s*ملی\s*[:：]?\s*(\d+)", t):
        if len(m.group(1)) != 10:
            warn.append(f"national ID with {len(m.group(1))} digits: {m.group(1)}")
    return warn


def checks(final: str, cross: str | None, layer: str, min_layer_chars: int = 30) -> dict:
    """Numbers not confirmed by the cross-read, [?] marks, format warnings and, with a text layer, recall stats."""
    fin = numbers(final)
    out = {"numbers": len(set(fin))}
    out["unconfirmed_by_cross_read"] = sorted(set(fin) - set(numbers(cross))) if cross is not None else []
    out["uncertain_marked"] = len(re.findall(r"\[\?\]", final))
    out["format_warnings"] = format_warnings(final)
    if len(norm(layer).strip()) > min_layer_chars:
        ln = set(numbers(layer))
        fw, lw = Counter(words(final)), Counter(words(layer))
        common = sum((fw & lw).values())
        out["text_layer"] = {
            "numbers_in_layer": len(ln),
            "layer_numbers_found": len(ln & set(fin)),
            "layer_numbers_missing": sorted(ln - set(fin)),
            "ocr_numbers_not_in_layer": sorted(set(fin) - ln),
            "word_recall": round(common / max(1, sum(lw.values())), 3),
            "word_precision": round(common / max(1, sum(fw.values())), 3),
        }
    return out
