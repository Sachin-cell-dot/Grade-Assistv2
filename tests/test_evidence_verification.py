from core.evidence_verification import POLICY_VERSION, verify_extraction
from core.models import VisionExtraction


def extraction(section_scores, reported=14, questions=None):
    sections = []
    for index, score in enumerate(section_scores, 1):
        sections.append({"identifier": f"S{index}", "visible_section_score": None if score is None else {"obtained": score}, "questions": questions or []})
    payload = {"sections": sections}
    if reported is not None:
        payload["worksheet_reported_score"] = {"obtained": reported, "maximum": 20}
    return VisionExtraction.model_validate(payload)


def test_complete_section_scores_matching_report_are_consistent():
    result = verify_extraction(extraction([3, 3, 2, 2, 4]))
    assert result.status == "consistent"
    assert result.evidence_summary.visible_section_score_sum == 14
    assert result.policy_version == POLICY_VERSION


def test_section_total_mismatch_routes_to_review_with_factual_explanation():
    result = verify_extraction(extraction([3, 3, 2, 2, 1]))
    check = next(check for check in result.checks if check.name == "section_total_comparison")
    assert result.status == "needs_review"
    assert "section_total_mismatch" in result.warnings
    assert check.explanation == "Visible section scores sum to 11, while the worksheet reports 14."


def test_incomplete_scores_do_not_claim_mismatch():
    result = verify_extraction(extraction([3, None, 2]))
    assert result.status == "insufficient_evidence"
    assert "section_scores_incomplete" in result.warnings
    assert "section_total_mismatch" not in result.warnings
    assert result.evidence_summary.visible_section_score_sum is None


def test_missing_reported_total_and_no_sections_are_insufficient():
    missing_total = verify_extraction(extraction([3], reported=None))
    no_sections = verify_extraction(VisionExtraction())
    assert missing_total.status == "insufficient_evidence"
    assert "reported_total_missing" in missing_total.warnings
    assert no_sections.status == "insufficient_evidence"
    assert "no_visible_sections" in no_sections.blockers


def test_question_coverage_does_not_turn_symbols_into_marks_or_mutate_evidence():
    source = VisionExtraction.model_validate({"student": {"name": "Rufina"}, "sections": [{"identifier": "A", "visible_section_score": {"obtained": 0}, "questions": [
        {"identifier": "1", "question_text": "Q1", "student_answer": {"visible_text": "wrong"}, "teacher_marking": {"visible_symbols": ["cross"]}},
        {"identifier": "2", "teacher_marking": {"visible_symbols": ["tick"], "visible_individual_score": {"obtained": 0}}},
    ]}], "worksheet_reported_score": {"obtained": 0, "maximum": 20}})
    before = source.model_dump(mode="json")
    result = verify_extraction(source)
    summary = result.evidence_summary
    assert summary.visible_questions == 2
    assert summary.questions_with_text == 1
    assert summary.questions_with_student_answer == 1
    assert summary.questions_with_teacher_symbol == 2
    assert summary.questions_with_individual_mark == 1
    assert source.model_dump(mode="json") == before
    assert source.sections[0].questions[0].teacher_marking.visible_individual_score is None
    assert source.sections[0].questions[1].teacher_marking.visible_individual_score.obtained == 0


def test_all_visible_individual_marks_mismatch_reported_total_requires_review():
    source = extraction([None], reported=10, questions=[
        {"identifier": "1", "teacher_marking": {"visible_individual_score": {"obtained": 2}}},
        {"identifier": "2", "teacher_marking": {"visible_individual_score": {"obtained": 2}}},
        {"identifier": "3", "teacher_marking": {"visible_individual_score": {"obtained": 2}}},
        {"identifier": "4", "teacher_marking": {"visible_individual_score": {"obtained": 2}}},
        {"identifier": "5", "teacher_marking": {"visible_individual_score": {"obtained": 0}}},
        {"identifier": "6", "teacher_marking": {"visible_individual_score": {"obtained": 3}}},
    ])
    result = verify_extraction(source)
    check = next(check for check in result.checks if check.reason == "individual_total_mismatch")
    assert result.status == "needs_review"
    assert result.evidence_summary.visible_individual_mark_sum == 11
    assert check.explanation == "Visible individual marks sum to 11, while the worksheet reports 10."


def test_matching_individual_marks_do_not_create_mismatch():
    source = extraction([None], reported=11, questions=[
        {"identifier": "1", "teacher_marking": {"visible_individual_score": {"obtained": 2}}},
        {"identifier": "2", "teacher_marking": {"visible_individual_score": {"obtained": 2}}},
        {"identifier": "3", "teacher_marking": {"visible_individual_score": {"obtained": 2}}},
        {"identifier": "4", "teacher_marking": {"visible_individual_score": {"obtained": 1}}},
        {"identifier": "6", "teacher_marking": {"visible_individual_score": {"obtained": 4}}},
    ])
    result = verify_extraction(source)
    assert result.evidence_summary.visible_individual_mark_sum == 11
    assert "individual_total_mismatch" not in result.warnings
