from pathlib import Path

from core.local_result_cache import LocalGroqResultCache, cache_key_for
from core.provenance import build_provenance
from services.groq_vision_service import (
    GroqQuestionPaperExtraction,
    GroqRubricExtraction,
    GroqVisionError,
    GroqWorksheetExtraction,
)
import ui.extraction_demo as extraction_demo


def test_document_type_dispatches_only_to_matching_groq_method(monkeypatch, tmp_path):
    image = tmp_path / "english.jpeg"
    image.write_bytes(b"image")
    calls = []

    class Settings:
        groq_vision_model = "test-model"

    class FakeGroq:
        settings = Settings()

        def extract_answer_sheet(self, path):
            calls.append("answer")
            return GroqWorksheetExtraction.model_validate({"student_name": "Student", "sections": [{"questions": [{"identifier": "1"}]}]})

        def extract_question_paper(self, path):
            calls.append("question")
            return GroqQuestionPaperExtraction.model_validate({"worksheet_title": "Question paper", "sections": [{"questions": [{"identifier": "1", "printed_text": "Printed question"}]}]})

        def extract_rubric(self, path):
            calls.append("rubric")
            return GroqRubricExtraction.model_validate({"worksheet_title": "Rubric", "sections": [{"questions": [{"identifier": "1", "visible_criteria": ["Visible criterion"]}]}]})

    monkeypatch.setattr(extraction_demo, "GroqVisionService", FakeGroq)
    extraction_demo.extract_uploaded_document(image, "question_paper")
    assert calls == ["question"]
    extraction_demo.extract_uploaded_document(image, "rubric")
    assert calls == ["question", "rubric"]
    extraction_demo.extract_uploaded_document(image, "student_answer_sheet")
    assert calls == ["question", "rubric", "answer"]
    assert "OllamaVisionService" not in extraction_demo.__dict__


def test_cache_key_includes_image_document_type_model_and_policy(tmp_path):
    image = tmp_path / "english.jpeg"
    image.write_bytes(b"image")
    base, fingerprint = cache_key_for(image, "student_answer_sheet", "model-a", "policy-a")
    assert fingerprint
    assert base != cache_key_for(image, "question_paper", "model-a", "policy-a")[0]
    assert base != cache_key_for(image, "student_answer_sheet", "model-b", "policy-a")[0]
    assert base != cache_key_for(image, "student_answer_sheet", "model-a", "policy-b")[0]
    image.write_bytes(b"different image")
    assert base != cache_key_for(image, "student_answer_sheet", "model-a", "policy-a")[0]


def test_cache_is_opt_in_and_cached_result_is_explicitly_labelled(tmp_path):
    image = tmp_path / "english.jpeg"
    image.write_bytes(b"image")
    cache = LocalGroqResultCache(tmp_path / "cache")
    provenance = build_provenance(image, provider="groq", model="model-a", policy_version="policy-a")
    assert cache.get(image, "student_answer_sheet", "model-a", "policy-a") is None
    cached = cache.put(image, "student_answer_sheet", "model-a", "policy-a", {"student": {}}, provenance)
    loaded = cache.get(image, "student_answer_sheet", "model-a", "policy-a")
    assert loaded is not None
    assert loaded.cache_key == cached.cache_key
    assert "Verified local result" in Path(extraction_demo.__file__).read_text(encoding="utf-8")
    assert "Run Groq again" in Path(extraction_demo.__file__).read_text(encoding="utf-8")


def test_429_has_calm_teacher_readable_message():
    message = extraction_demo.teacher_readable_groq_error(GroqVisionError("Groq Vision returned HTTP 429."))
    assert message == "Groq is temporarily rate-limited. Please wait briefly, then retry."
