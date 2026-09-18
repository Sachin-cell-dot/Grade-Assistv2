from core.models import Question, VisionExtraction
from core.partial_credit import (
    LexicalConceptConfig,
    SuggestionStatus,
    TeacherDispositionAction,
    append_teacher_disposition,
    partial_credit_suggestions,
    record_teacher_disposition,
    suggest_partial_credit,
)
from core.reference_models import RubricExtraction, RubricQuestion


def question(answer: str | None, teacher_mark: float | None = None, identifier: str = "1"):
    payload = {"identifier": identifier, "student_answer": {"visible_text": answer}}
    if teacher_mark is not None:
        payload["teacher_marking"] = {"visible_individual_score": {"obtained": teacher_mark}}
    return Question.model_validate(payload)


def rubric(criteria, maximum=4, identifier="1"):
    return RubricQuestion.model_validate({"identifier": identifier, "visible_criteria": criteria, "visible_maximum_marks": maximum})


def test_full_concept_coverage_agrees_with_visible_teacher_mark():
    suggestion = suggest_partial_credit(
        question("The blue bird sat in a tall tree.", teacher_mark=4),
        rubric(["identifies blue bird", "mentions tall tree"], maximum=4),
    )
    assert suggestion.suggested_mark == 4
    assert suggestion.status == SuggestionStatus.AGREES_WITH_TEACHER
    assert len(suggestion.matched_rubric_criteria) == 2
    assert suggestion.evidence_phrases == ["The blue bird sat in a tall tree."]


def test_partial_weighted_long_answer_coverage_requires_review():
    suggestion = suggest_partial_credit(
        question("The response discusses clean energy.", teacher_mark=0),
        rubric(["mentions clean energy", "explains reduced pollution"], maximum=5),
    )
    assert suggestion.suggested_mark == 2.5
    assert suggestion.status == SuggestionStatus.SUGGEST_REVIEW
    assert len(suggestion.missing_rubric_criteria) == 1


def test_missing_answer_or_rubric_is_insufficient_evidence():
    assert suggest_partial_credit(question(None), rubric(["visible criterion"])).status == SuggestionStatus.INSUFFICIENT_EVIDENCE
    assert suggest_partial_credit(question("Visible answer"), None).status == SuggestionStatus.INSUFFICIENT_EVIDENCE


def test_suggestion_is_capped_and_never_mutates_teacher_evidence():
    source = question("helpful support aid", teacher_mark=1)
    before = source.model_dump(mode="json")
    suggestion = suggest_partial_credit(source, rubric(["help", "support", "aid"], maximum=2))
    assert suggestion.suggested_mark <= suggestion.rubric_maximum_marks
    assert source.model_dump(mode="json") == before
    assert source.teacher_marking.visible_individual_score.obtained == 1


def test_matching_mark_stays_quiet_and_differing_mark_requires_action():
    matching = suggest_partial_credit(question("blue bird", teacher_mark=2), rubric(["blue bird"], maximum=2))
    differing = suggest_partial_credit(question("blue bird", teacher_mark=0), rubric(["blue bird"], maximum=2))
    assert matching.status == SuggestionStatus.AGREES_WITH_TEACHER
    assert differing.status == SuggestionStatus.SUGGEST_REVIEW
    disposition = record_teacher_disposition(differing, TeacherDispositionAction.ACCEPT_SUGGESTION, "Teacher accepted visible rubric comparison")
    assert disposition.original_teacher_mark == 0
    assert disposition.system_suggested_mark == 2
    assert disposition.teacher_final_mark == 2


def test_teacher_dispositions_are_append_only():
    suggestion = suggest_partial_credit(question("blue bird", teacher_mark=0), rubric(["blue bird"], maximum=2))
    first = record_teacher_disposition(suggestion, TeacherDispositionAction.EDIT_FINAL_MARK, edited_final_mark=1)
    history = append_teacher_disposition([], first)
    second = record_teacher_disposition(suggestion, TeacherDispositionAction.PROCEED_WITH_TEACHER_MARK)
    expanded = append_teacher_disposition(history, second)
    assert len(history) == 1
    assert len(expanded) == 2
    assert expanded[0] == first


def test_rubric_matching_uses_only_answer_and_rubric_not_question_paper_or_groq():
    answer_sheet = VisionExtraction.model_validate({"sections": [{"questions": [{"identifier": "Q1", "student_answer": {"visible_text": "blue bird"}}]}]})
    rubric_sheet = RubricExtraction.model_validate({"sections": [{"questions": [{"identifier": "1", "visible_criteria": ["blue bird"], "visible_maximum_marks": 2}]}]})
    suggestions = partial_credit_suggestions(answer_sheet, rubric_sheet, LexicalConceptConfig(semantic_threshold=0.5))
    assert suggestions[0].suggested_mark == 2
    production = open("core/partial_credit.py", encoding="utf-8").read()
    assert "GroqVisionService" not in production
    assert "Meera" not in production
