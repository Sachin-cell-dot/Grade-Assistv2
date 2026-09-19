import json

import httpx
import pytest

from config.settings import Settings
from core.models import VisionExtraction
from services.groq_vision_service import (
    GROQ_CHAT_COMPLETIONS_URL,
    PROMPT,
    GroqVisionError,
    GroqVisionExtractionError,
    GroqVisionService,
    GroqWorksheetExtraction,
    groq_dto_to_vision_extraction,
)
from tools.scoring import deterministic_total


def make_service(handler):
    return GroqVisionService(Settings(groq_api_key="test-groq-secret", groq_timeout_seconds=5), httpx.Client(transport=httpx.MockTransport(handler)))


def image_file(tmp_path):
    path = tmp_path / "worksheet.jpeg"
    path.write_bytes(b"image-bytes")
    return path


def response(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def rich_payload():
    return {
        "student_name": "Rufina Thomas", "class_name": "Class 2", "subject": "Mathematics", "assessment_date": "19/09/2026",
        "reported_score_obtained": 14, "reported_score_maximum": 20, "teacher_comment": "Should improve",
        "sections": [{"identifier": "Addition", "reported_score": 3, "questions": [
            {"identifier": "1", "question_text": "1 + 1", "student_answer": "3", "teacher_symbol": "cross"},
            {"identifier": "2", "question_text": "2 + 2", "student_answer": "4", "teacher_symbol": "tick"},
        ]}],
        "unassigned_teacher_annotations": [],
    }


def test_plain_json_request_parses_without_response_format(tmp_path):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return response(json.dumps(rich_payload()))

    dto = make_service(handler).extract_image(image_file(tmp_path))
    assert seen["url"] == GROQ_CHAT_COMPLETIONS_URL
    assert "response_format" not in seen["body"]
    assert seen["body"]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert dto.student_name == "Rufina Thomas"


@pytest.mark.parametrize(
    ("method", "payload", "setting_name", "budget"),
    [
        ("extract_answer_sheet", rich_payload(), "groq_answer_sheet_max_completion_tokens", 2201),
        ("extract_question_paper", {"worksheet_title": "Paper", "sections": []}, "groq_question_paper_max_completion_tokens", 3001),
        ("extract_rubric", {"worksheet_title": "Rubric", "sections": []}, "groq_rubric_max_completion_tokens", 2202),
    ],
)
def test_document_type_uses_its_configured_completion_budget(tmp_path, method, payload, setting_name, budget):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return response(json.dumps(payload))

    settings = Settings(groq_api_key="test-groq-secret", **{setting_name: budget})
    service = GroqVisionService(settings, httpx.Client(transport=httpx.MockTransport(handler)))
    getattr(service, method)(image_file(tmp_path))
    assert seen["max_completion_tokens"] == budget


@pytest.mark.parametrize("setting_name", ["groq_answer_sheet_max_completion_tokens", "groq_question_paper_max_completion_tokens", "groq_rubric_max_completion_tokens"])
def test_completion_budget_must_be_positive_reasonable_integer(setting_name):
    with pytest.raises(ValueError):
        Settings(**{setting_name: 0})


def test_optional_json_fence_strips_safely(tmp_path):
    fenced = "```json\n" + json.dumps(rich_payload()) + "\n```"
    assert make_service(lambda request: response(fenced)).extract_image(image_file(tmp_path)).sections[0].identifier == "Addition"


def test_malformed_json_and_empty_extraction_fail_visibly(tmp_path):
    with pytest.raises(GroqVisionError, match="malformed JSON"):
        make_service(lambda request: response("not json")).extract_image(image_file(tmp_path))
    with pytest.raises(GroqVisionExtractionError, match="semantically empty"):
        make_service(lambda request: response(json.dumps({"sections": [], "unassigned_teacher_annotations": []}))).extract_image(image_file(tmp_path))


def test_partial_legitimate_extraction_is_allowed(tmp_path):
    dto = make_service(lambda request: response(json.dumps({"student_name": "Rufina Thomas", "sections": []}))).extract_image(image_file(tmp_path))
    assert dto.student_name == "Rufina Thomas"


def test_dto_to_canonical_preserves_evidence_without_numeric_inference(tmp_path):
    dto = make_service(lambda request: response(json.dumps(rich_payload()))).extract_image(image_file(tmp_path))
    result = groq_dto_to_vision_extraction(dto)
    first, second = result.sections[0].questions
    assert isinstance(result, VisionExtraction)
    assert first.question_text == "1 + 1"
    assert first.student_answer.visible_text == "3"
    assert first.teacher_marking.visible_symbols == ["cross"]
    assert second.teacher_marking.visible_symbols == ["tick"]
    assert first.teacher_marking.visible_individual_score is None
    assert result.sections[0].visible_section_score.obtained == 3
    assert result.worksheet_reported_score.obtained == 14
    assert result.worksheet_reported_score.maximum == 20
    assert result.unassigned_teacher_markings[0].visible_comment == "Should improve"
    assert deterministic_total(result) is None


def test_answer_sheet_duplicate_answer_is_not_presented_as_question_text():
    dto = GroqWorksheetExtraction.model_validate({
        "sections": [{"questions": [{
            "identifier": "1", "question_text": "Student's handwritten answer",
            "student_answer": "Student's handwritten answer",
        }]}],
    })
    question = groq_dto_to_vision_extraction(dto).sections[0].questions[0]
    assert question.question_text is None
    assert question.student_answer.visible_text == "Student's handwritten answer"


def test_answer_sheet_prompt_requires_visible_only_question_text_and_no_reference_use():
    assert "question_text` means text visibly printed on THIS answer-sheet image only" in PROMPT
    assert "Never copy `student_answer`" in PROMPT
    assert "Do not use a question paper, rubric, answer key" in PROMPT


def test_visible_scores_and_teacher_evidence_are_preserved_without_inference():
    dto = GroqWorksheetExtraction.model_validate({
        "reported_score_obtained": 10, "reported_score_maximum": 20,
        "sections": [{"reported_score": None, "questions": [{
            "identifier": "1", "student_answer": "visible answer", "teacher_symbol": "tick",
            "teacher_correction": "red correction", "awarded_mark": 1,
        }]}],
    })
    result = groq_dto_to_vision_extraction(dto)
    question = result.sections[0].questions[0]
    assert result.worksheet_reported_score.obtained == 10
    assert result.worksheet_reported_score.maximum == 20
    assert result.sections[0].visible_section_score is None
    assert question.teacher_marking.visible_individual_score.obtained == 1
    assert question.teacher_marking.visible_symbols == ["tick"]
    assert question.teacher_marking.visible_correction == "red correction"


def test_placeholder_key_fails_locally_without_request(tmp_path):
    service = GroqVisionService(Settings(groq_api_key="replace_me"))
    with pytest.raises(GroqVisionError, match="missing, blank, or a placeholder"):
        service.extract_image(image_file(tmp_path))


def test_separate_student_dtos_do_not_leak_evidence_between_extractions():
    sachin = GroqWorksheetExtraction.model_validate({"student_name": "Sachin", "sections": [{"questions": [{"identifier": "1", "student_answer": "Sachin answer", "awarded_mark": 1}]}]})
    rufina = GroqWorksheetExtraction.model_validate({"student_name": "Rufina", "sections": [{"questions": [{"identifier": "1", "student_answer": "Rufina answer", "teacher_symbol": "cross"}]}]})
    sachin_result = groq_dto_to_vision_extraction(sachin)
    rufina_result = groq_dto_to_vision_extraction(rufina)
    assert sachin_result.student.name == "Sachin"
    assert sachin_result.sections[0].questions[0].teacher_marking.visible_individual_score.obtained == 1
    assert rufina_result.student.name == "Rufina"
    assert rufina_result.sections[0].questions[0].student_answer.visible_text == "Rufina answer"
    assert rufina_result.sections[0].questions[0].teacher_marking.visible_symbols == ["cross"]
