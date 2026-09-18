from core.models import VisionExtraction
from tools.scoring import deterministic_total
from services.ollama_service import PROMPT


def test_canonical_schema_supports_full_page_evidence_and_null_identifiers():
    result = VisionExtraction.model_validate({"sections": [{"identifier": None, "questions": [{"identifier": None, "question_text": "Complete question", "student_answer": {"visible_text": "Answer", "visible_working": "Working"}, "teacher_marking": {"visible_correction": "Visible correction", "visible_individual_score": {"obtained": 1}}}]}], "worksheet_reported_score": {"obtained": 1}})
    question = result.sections[0].questions[0]
    assert question.identifier is None
    assert question.question_text == "Complete question"
    assert question.teacher_marking.visible_correction == "Visible correction"


def test_prompt_is_transcription_only_and_handles_ambiguous_annotations():
    lowered = PROMPT.lower()
    for prohibited in ("never solve", "correct a student answer", "judge correctness", "derive marks", "do not calculate totals"):
        assert prohibited in lowered
    assert "unassigned teacher marking" in lowered
    assert "do not invent identifiers" in lowered


def test_handwritten_teacher_score_comment_and_red_marks_are_preserved():
    result = VisionExtraction.model_validate({"worksheet_reported_score": {"obtained": 14, "maximum": 20}, "unassigned_teacher_markings": [{"visible_comment": "Should Improve"}], "sections": [{"identifier": "Fractions", "visible_section_score": None, "questions": [{"identifier": "1", "teacher_marking": {"visible_symbols": ["red tick", "red circle"], "visible_individual_score": {"obtained": 2}}}, {"identifier": "2", "teacher_marking": {"visible_symbols": ["red cross"], "visible_individual_score": {"obtained": 0}}}]}]})
    assert result.worksheet_reported_score.obtained == 14
    assert result.worksheet_reported_score.maximum == 20
    assert result.unassigned_teacher_markings[0].visible_comment == "Should Improve"
    assert result.sections[0].questions[1].teacher_marking.visible_symbols == ["red cross"]


def test_visible_parent_evidence_survives_missing_nested_evidence():
    result = VisionExtraction.model_validate({"sections": [{"identifier": "Geometry", "visible_section_score": None, "questions": [{"identifier": None, "question_text": "Visible question", "teacher_marking": None}]}]})
    assert len(result.sections) == 1
    assert result.sections[0].visible_section_score is None
    assert result.sections[0].questions[0].teacher_marking is None
    assert result.sections[0].questions[0].identifier is None


def test_unclear_evidence_requires_uncertainty_and_missing_marks_are_not_inferred():
    valid = VisionExtraction.model_validate({"sections": [{"identifier": "A", "questions": [{"identifier": "1", "student_answer": {"visible_text": "12"}, "teacher_marking": {"uncertainty": {"is_uncertain": True, "source_legibility": "unclear", "reasons": ["mark unreadable"]}}}]}]})
    assert deterministic_total(valid) is None
    try:
        VisionExtraction.model_validate({"overall_uncertainty": {"is_uncertain": False, "source_legibility": "unclear"}})
    except ValueError:
        pass
    else:
        raise AssertionError("Unclear evidence must be uncertain")
