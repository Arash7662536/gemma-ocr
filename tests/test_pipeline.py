"""End-to-end run against a fake Gemma: a born-digital PDF and a scanned image go through all passes."""
import json

import pymupdf
from PIL import Image, ImageDraw

from gemma_ocr import pipeline
from gemma_ocr.audit import audit
from gemma_ocr.client import GemmaClient
from gemma_ocr.report import build_report

from .conftest import FakeGemma


def make_inputs(d):
    d.mkdir()
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Case 140268920007597145 date 1403/10/23 some text layer content here", fontsize=11)
    doc.save(d / "digital.pdf")
    img = Image.new("RGB", (1200, 1600), "white")
    ImageDraw.Draw(img).rectangle((100, 100, 1100, 1500), outline="black", width=5)
    img.save(d / "scan.jpg")
    img.save(d / "scan_copy.jpg")          # identical bytes -> deduplicated
    (d / "._junk.pdf").write_bytes(b"x")   # macOS resource fork -> skipped


def test_run_end_to_end(cfg, tmp_path, monkeypatch):
    make_inputs(tmp_path / "in")
    fake = FakeGemma(transcript="شماره پرونده : 140268920007597146\nتاریخ : 1403/10/23")
    monkeypatch.setattr(pipeline, "GemmaClient", lambda c: GemmaClient(c, session=fake))
    out = tmp_path / "out"
    m = pipeline.run(cfg, tmp_path / "in", out)
    assert m["unique_files"] == 2 and m["files"] == 3 and m["pages_ok"] == 2 and not m["failures"]
    assert json.loads((out / "duplicates.json").read_text()) == {"scan_copy.jpg": "scan.jpg"}

    pdf = json.loads((out / "pages" / "digital.pdf__p1.json").read_text())
    scan = json.loads((out / "pages" / "scan.jpg__p1.json").read_text())
    # designed branch: text-layer pages skip orientation, get the layer in the proofread and snapping
    assert pdf["has_text_layer"] and pdf["orientation_votes"] is None
    assert pdf["text_layer_fixes"] == [{"ocr": "140268920007597146", "text_layer": "140268920007597145"}]
    assert "140268920007597145" in pdf["text"]
    assert not scan["has_text_layer"] and scan["orientation_votes"] == [0, 180] and scan["rotation_ccw"] == 0
    assert scan["strips"] >= 2
    for r in (pdf, scan):
        assert set(r["passes"]) == {"draft", "proofread", "cross_read"}
        assert r["passes"]["draft"]["think"] and r["passes"]["proofread"]["think"]
        assert not r["passes"]["cross_read"]["think"]
        assert r["proofread_used"]
    proof_prompts = [b["messages"][0]["content"][-1]["text"] for b in fake.bodies if "proofreading" in b["messages"][0]["content"][-1]["text"]]
    assert sum("DIGITAL TEXT LAYER" in p for p in proof_prompts) == 1

    doc = json.loads((out / "docs" / "digital.pdf.json").read_text())
    assert doc["fields"]["document_type_fa"] == "ابلاغیه"

    n_calls = len(fake.bodies)
    pipeline.run(cfg, tmp_path / "in", out)        # rerun: everything from cache
    assert len(fake.bodies) == n_calls

    a = audit(out)
    assert a["pages"] == 2 and len(a["versions"]) == 1 and a["field_key_sets"] == 1
    html = build_report(out, "Test").read_text()
    assert "<h1>Test</h1>" in html and "scan.jpg" in html
    assert (out / "fields.csv").read_text(encoding="utf-8-sig").startswith("file,pages,case_group")


def test_orientation_rotates_only_when_votes_agree(cfg, tmp_path, monkeypatch):
    make_inputs(tmp_path / "in")
    fake = FakeGemma(orient_letter="B")   # order 1: B=90, order 2: B=270 -> disagree -> no rotation
    monkeypatch.setattr(pipeline, "GemmaClient", lambda c: GemmaClient(c, session=fake))
    pipeline.run(cfg, tmp_path / "in", tmp_path / "out", only=["scan.jpg"])
    r = json.loads((tmp_path / "out" / "pages" / "scan.jpg__p1.json").read_text())
    assert r["orientation_votes"] == [90, 270] and r["rotation_ccw"] == 0


def test_passes_can_be_disabled(cfg, tmp_path, monkeypatch):
    make_inputs(tmp_path / "in")
    cfg.proofread = cfg.cross_read = cfg.fields = cfg.orientation = False
    fake = FakeGemma()
    monkeypatch.setattr(pipeline, "GemmaClient", lambda c: GemmaClient(c, session=fake))
    pipeline.run(cfg, tmp_path / "in", tmp_path / "out", only=["scan.jpg"])
    r = json.loads((tmp_path / "out" / "pages" / "scan.jpg__p1.json").read_text())
    assert set(r["passes"]) == {"draft"} and not r["proofread_used"] and len(fake.bodies) == 1
