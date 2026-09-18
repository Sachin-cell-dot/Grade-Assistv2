import json

import httpx

from config.settings import Settings
from core.reference_models import QuestionPaperExtraction, RubricExtraction
from services.groq_vision_service import (
    QUESTION_PAPER_PROMPT,
    RUBRIC_PROMPT,
    GroqVisionService,
    groq_question_paper_to_extraction,
    groq_rubric_to_extraction,
)


def _service(handler):
    return GroqVisionService(Settings(groq_api_key="test-groq-secret"), httpx.Client(transport=httpx.MockTransport(handler)))


def _image(tmp_path):
    image = tmp_path / "reference.jpeg"
    image.write_bytes(b"reference-image")
    return image


def _response(payload):
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}}]})


def test_question_paper_uses_its_own_prompt_and_typed_printed_evidence(tmp_path):
    seen = {}

    def handler(request):
        seen["prompt"] = json.loads(request.content)["messages"][0]["content"][0]["text"]
        return _response({"worksheet_title": "English", "subject": "English", "total_marks": 20, "sections": [{"identifier": "Part A", "printed_maximum_marks": 10, "questions": [{"identifier": "1", "printed_text": "Why was the man worried?", "printed_maximum_marks": 2}]}]})

    result = groq_question_paper_to_extraction(_service(handler).extract_question_paper(_image(tmp_path)))
    assert isinstance(result, QuestionPaperExtraction)
    assert result.sections[0].questions[0].printed_text == "Why was the man worried?"
    assert result.sections[0].questions[0].printed_maximum_marks == 2
    assert seen["prompt"] == QUESTION_PAPER_PROMPT
    assert "create answers" in seen["prompt"]


def test_rubric_uses_its_own_prompt_and_never_applies_criteria(tmp_path):
    seen = {}

    def handler(request):
        seen["prompt"] = json.loads(request.content)["messages"][0]["content"][0]["text"]
        return _response({"worksheet_title": "Rubrics", "subject": "English", "sections": [{"title": "Part A", "questions": [{"identifier": "1", "visible_criteria": ["Identifies worry"], "visible_maximum_marks": 2, "visible_key_context": "Heavy rain"}]}]})

    result = groq_rubric_to_extraction(_service(handler).extract_rubric(_image(tmp_path)))
    assert isinstance(result, RubricExtraction)
    assert result.sections[0].questions[0].visible_criteria == ["Identifies worry"]
    assert result.sections[0].questions[0].visible_maximum_marks == 2
    assert seen["prompt"] == RUBRIC_PROMPT
    assert "Do not grade a student" in seen["prompt"]
    assert "alter an answer sheet" in seen["prompt"]


def test_reference_prompts_are_not_answer_sheet_prompts():
    assert "student_answer" not in QUESTION_PAPER_PROMPT
    assert "student_answer" not in RUBRIC_PROMPT
    assert "question_text" not in QUESTION_PAPER_PROMPT
