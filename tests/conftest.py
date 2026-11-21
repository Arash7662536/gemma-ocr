import json

import pytest

from gemma_ocr.config import Config


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeGemma:
    """Stands in for requests.Session: answers by looking at the prompt, records every request body."""

    def __init__(self, transcript="شماره پرونده : ۱۴۰۲۶۸۹۲۰۰۰۷۵۹۷۱۴۵\nتاریخ : ۱۴۰۳/۱۰/۲۳", orient_letter="A",
                 loop_thinking=False):
        self.transcript = transcript
        self.orient_letter = orient_letter
        self.loop_thinking = loop_thinking
        self.bodies = []

    def post(self, url, json=None, timeout=None, headers=None):
        self.bodies.append(json)
        prompt = json["messages"][0]["content"][-1]["text"]
        think = json["chat_template_kwargs"]["enable_thinking"]
        if self.loop_thinking and think and "OCR transcriber" in prompt:
            return FakeResponse(self._msg("", "length"))
        if "rotated in different ways" in prompt:
            return FakeResponse(self._msg(self.orient_letter))
        if "Extract structured information" in prompt:
            return FakeResponse(self._msg(__import__("json").dumps({"document_type_fa": "ابلاغیه", "case_number": "1"})))
        return FakeResponse(self._msg(self.transcript))

    @staticmethod
    def _msg(text, finish="stop"):
        return {"choices": [{"message": {"content": text}, "finish_reason": finish}], "usage": {"prompt_tokens": 1}}


@pytest.fixture
def cfg(tmp_path):
    return Config(cache_dir=str(tmp_path / "cache"), workers=2, api_key="test")
