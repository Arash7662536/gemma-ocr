import json

import pytest

from gemma_ocr import prompts
from gemma_ocr.client import CallResult, GemmaClient, clean, looks_bad
from gemma_ocr.config import Config, load_config

from .conftest import FakeGemma


def test_prompt_templates_format():
    p = prompts.TRANSCRIBE.format(doc_context="X", images_note=prompts.images_note(3))
    assert "Images 2-4 are 3 overlapping" in p
    prompts.PROOFREAD.format(doc_context="X", images_note="n", draft="{not a field}", layer_note="")
    f = prompts.FIELDS.format(doc_context="X", name="a.pdf", text="t")
    assert '"summary_fa"' in f and "{\n" in f
    for k in prompts.FIELD_KEYS:
        assert f'"{k}"' in f


def test_clean():
    assert clean("```markdown\nسلام\n```") == "سلام"
    assert clean("<<<\nسلام\n>>>") == "سلام"


def test_looks_bad():
    assert looks_bad(CallResult("", "stop", True))
    assert looks_bad(CallResult("x" * 100, "length", True))
    assert looks_bad(CallResult("\n".join(["same line"] * 40), "stop", False))
    assert not looks_bad(CallResult("a normal transcription text", "stop", False))


def test_request_body_and_cache(cfg):
    fake = FakeGemma()
    c = GemmaClient(cfg, session=fake)
    content = [{"type": "text", "text": "hello OCR transcriber"}]
    r1 = c.call(content, think=True, max_tokens=10, tag="t_")
    r2 = c.call(content, think=True, max_tokens=10, tag="t_")
    assert len(fake.bodies) == 1 and not r1.cached and r2.cached and r1.text == r2.text
    b = fake.bodies[0]
    assert b["temperature"] == 0.0 and isinstance(b["temperature"], float)
    assert b["mm_processor_kwargs"] == {"max_soft_tokens": 1120}
    assert b["chat_template_kwargs"] == {"enable_thinking": True}
    assert b["repetition_penalty"] == 1.03


def test_call_robust_falls_back(cfg):
    fake = FakeGemma(loop_thinking=True)
    c = GemmaClient(cfg, session=fake)
    r = c.call_robust([{"type": "text", "text": "OCR transcriber"}], think_first=True, max_tokens=10, tag="t_")
    assert r.fallback and not r.think and r.finish == "stop"


def test_config_load_order(tmp_path, monkeypatch):
    toml = tmp_path / "c.toml"
    toml.write_text('dpi = 200\nproofread = false\n')
    monkeypatch.setenv("GEMMA_KEY", "k")
    monkeypatch.chdir(tmp_path)
    c = load_config(str(toml), ["dpi=250", "strip_overlap=0.2"])
    assert (c.dpi, c.proofread, c.api_key, c.strip_overlap) == (250, False, "k", 0.2)
    with pytest.raises(KeyError):
        c.set("nope", "1")
    assert "api_key" not in c.public()


def test_fingerprint_ignores_runtime_settings():
    a, b = Config(workers=1, api_key="x"), Config(workers=32, api_key="y")
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != Config(dpi=200).fingerprint()
