import json

import httpx
import pytest

from config.settings import Settings
from core.models import VisionExtraction
from services.groq_vision_service import (
    GROQ_CHAT_COMPLETIONS_URL,
    GroqVisionError,
    GroqVisionExtractionError,
    GroqVisionService,
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
