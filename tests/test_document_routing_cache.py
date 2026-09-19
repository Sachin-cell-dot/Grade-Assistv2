from pathlib import Path
import json

from core.local_result_cache import LocalGroqResultCache, cache_key_for
from core.provenance import build_provenance
from core.teacher_evidence_import import import_teacher_verified_answer_evidence
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


def test_imported_cache_can_be_found_by_explicit_source_image_name(tmp_path):
    source = tmp_path / "prior.json"
    source.write_bytes(b"evidence")
    cache = LocalGroqResultCache(tmp_path / "cache")
    provenance = build_provenance(source, provider="teacher_verified_import", model="teacher-local", policy_version="policy-a")
    provenance = provenance.model_copy(update={"source_image_identifier": "student.jpeg", "source_type": "teacher_verified_import"})
    cache.put(source, "student_answer_sheet", "teacher-local", "policy-a", {"student": {}}, provenance)
    found = cache.for_source_image_identifier("student_answer_sheet", "student.jpeg", "policy-a")
    assert found is not None
    assert found.provenance.model == "teacher-local"


def test_teacher_verified_answer_evidence_is_discoverable_by_original_upload_filename(tmp_path):
    evidence = tmp_path / "rufina.json"
    evidence.write_text(json.dumps({"sections": [{"questions": [{"identifier": "Q1", "student_answer": {"visible_text": "Visible answer"}}]}]}), encoding="utf-8")
    upload = tmp_path / "Rufina_Answersheet-content-fingerprint.jpeg"
    upload.write_bytes(b"worksheet-image")
    cache = LocalGroqResultCache(tmp_path / "cache")
    imported = import_teacher_verified_answer_evidence(
        evidence,
        source_image_filename="Rufina_Answersheet.jpeg",
        policy_version=extraction_demo.POLICY_VERSION,
        cache=cache,
    )

    found = extraction_demo.matching_local_result(
        cache,
        upload,
        "student_answer_sheet",
        "qwen/qwen3.8-27b",
        "Rufina_Answersheet.jpeg",
    )

    assert found is not None
    assert found.cache_key == imported.cache_key
    assert found.provenance.source_type == "teacher_verified_import"


def test_429_has_calm_teacher_readable_message():
    message = extraction_demo.teacher_readable_groq_error(GroqVisionError("Groq Vision returned HTTP 429."))
    assert message == "Groq is temporarily rate-limited. Please wait briefly, then retry."


def test_document_session_state_is_namespaced_and_replacement_is_isolated(tmp_path):
    session = {}
    answer = extraction_demo.document_state_for(session, "student_answer_sheet")
    paper = extraction_demo.document_state_for(session, "question_paper")
    rubric = extraction_demo.document_state_for(session, "rubric")
    answer_image = tmp_path / "answer.jpeg"
    paper_image = tmp_path / "paper.jpeg"
    rubric_image = tmp_path / "rubric.jpeg"
    replacement_paper = tmp_path / "paper-replacement.jpeg"

    assert extraction_demo.store_uploaded_document(answer, source_path=answer_image, image_path=answer_image, source_filename="answer.jpeg", label="Answer") is True
    answer.update({"result": {"student": {"name": "Student"}}, "source_label": "Recovered prior Groq evidence"})
    assert extraction_demo.store_uploaded_document(paper, source_path=paper_image, image_path=paper_image, source_filename="paper.jpeg", label="Paper") is True
    paper.update({"result": {"sections": []}, "provenance": {"model": "model-a"}, "source_label": "Verified local result"})
    assert extraction_demo.store_uploaded_document(rubric, source_path=rubric_image, image_path=rubric_image, source_filename="rubric.jpeg", label="Rubric") is True
    rubric.update({"result": {"sections": []}, "source_label": "Verified local result"})

    # Returning to a type reads its own retained state rather than a global latest result.
    assert extraction_demo.document_state_for(session, "question_paper")["result"] == {"sections": []}
    assert extraction_demo.document_state_for(session, "student_answer_sheet")["source_label"] == "Recovered prior Groq evidence"
    assert extraction_demo.document_state_for(session, "rubric")["source_label"] == "Verified local result"

    # A new question paper clears only that paper's result/review state.
    assert extraction_demo.store_uploaded_document(paper, source_path=replacement_paper, image_path=replacement_paper, source_filename="new-paper.jpeg", label="New paper") is True
    assert "result" not in paper
    assert answer["result"] == {"student": {"name": "Student"}}
    assert rubric["result"] == {"sections": []}


def test_teacher_facing_evidence_rows_hide_internal_extraction_fields():
    from core.models import VisionExtraction

    extraction = VisionExtraction.model_validate({
        "sections": [{"questions": [{
            "identifier": "Q1",
            "student_answer": {"visible_text": "Visible answer", "visible_working": "Hidden working"},
            "teacher_marking": {"visible_symbols": [], "visible_individual_score": {"obtained": 2}},
        }]}],
    })

    assert extraction_demo.answer_evidence_rows(extraction) == [{
        "Question": "Q1",
        "Student answer": "Visible answer",
        "Teacher mark": 2.0,
        "Max mark": None,
    }]
    source = Path(extraction_demo.__file__).read_text(encoding="utf-8")
    assert source.count("st.json(") == 1
    assert "Technical evidence / raw extraction" in source


def test_worksheet_reported_total_display_uses_obtained_value_not_visiblescore():
    from core.models import VisionExtraction

    present = VisionExtraction.model_validate({"worksheet_reported_score": {"obtained": 10, "maximum": 20}})
    absent = VisionExtraction.model_validate({})

    assert extraction_demo.worksheet_reported_total_display(present) == 10.0
    assert extraction_demo.worksheet_reported_total_display(absent) == "Not visible"
