import pytest

from core.evidence_verification import verify_extraction
from core.models import VisionExtraction
from core.routing import ReviewRoute, route_extraction
from core.teacher_review import (
    ReviewAction,
    TeacherCorrection,
    apply_correction_overlay,
    correction_value,
    resolve_teacher_review,
)


def extracted(reported=10, marks=(2, 2, 2, 2, 0, 3), section_score=None):
    return VisionExtraction.model_validate({
        "sections": [{"identifier": "Part A", "visible_section_score": section_score, "questions": [
            {"identifier": str(index), "teacher_marking": {"visible_individual_score": {"obtained": mark}}}
            for index, mark in enumerate(marks, 1)
        ]}],
        "worksheet_reported_score": {"obtained": reported, "maximum": 20},
    })


def test_mismatch_routes_teacher_review_with_factual_item():
    source = extracted()
    decision = route_extraction(source, verify_extraction(source))
    assert decision.route == ReviewRoute.TEACHER_REVIEW
    assert decision.route != ReviewRoute.BLOCKED
    assert decision.reason_codes == ["individual_total_mismatch"]
    assert decision.review_items[0].summary == "Visible individual marks sum to 11, while the worksheet reports 10."


def test_rufina_style_matching_evidence_auto_continues_even_without_section_score():
    source = extracted(reported=11, marks=(2, 2, 2, 1, 4), section_score=None)
    decision = route_extraction(source, verify_extraction(source))
    assert decision.route == ReviewRoute.TEACHER_REVIEW
    assert "individual_total_mismatch" not in decision.reason_codes
    assert "section_scores_incomplete" in decision.reason_codes


def test_complete_consistent_evidence_auto_continues():
    source = extracted(reported=11, marks=(2, 2, 2, 1, 4), section_score={"obtained": 11})
    assert route_extraction(source, verify_extraction(source)).route == ReviewRoute.AUTO_CONTINUE


def test_overlay_preserves_original_and_reverifies_corrected_evidence():
    source = extracted()
    original = source.model_dump(mode="json")
    path = "worksheet_reported_score.obtained"
    correction = TeacherCorrection(review_item_id="item-1", field_path=path, original_value=correction_value(source, path), corrected_value=11, teacher_note="Confirmed written total")
    effective = apply_correction_overlay(source, [correction])
    assert source.model_dump(mode="json") == original
    assert effective.worksheet_reported_score.obtained == 11
    resolution = resolve_teacher_review(source, ReviewAction.CORRECT_EVIDENCE, [correction])
    assert resolution.verification.status == "insufficient_evidence"
    assert "individual_total_mismatch" not in resolution.verification.warnings
    assert resolution.routing.route == ReviewRoute.TEACHER_REVIEW


def test_defer_does_not_verify_or_change_evidence():
    source = extracted()
    resolution = resolve_teacher_review(source, ReviewAction.DEFER)
    assert resolution.verification is None
    assert resolution.routing is None
    assert resolution.effective_extraction.model_dump(mode="json") == source.model_dump(mode="json")


def test_invalid_correction_path_and_type_are_rejected():
    with pytest.raises(ValueError, match="not supported"):
        TeacherCorrection(review_item_id="item", field_path="student.name", original_value=1, corrected_value=2)
    with pytest.raises(ValueError):
        TeacherCorrection.model_validate({"review_item_id": "item", "field_path": "worksheet_reported_score.obtained", "original_value": 1, "corrected_value": "not-a-number"})


def test_no_question_evidence_is_blocked_without_grading_logic():
    source = VisionExtraction.model_validate({"worksheet_reported_score": {"obtained": 10}})
    decision = route_extraction(source, verify_extraction(source))
    assert decision.route == ReviewRoute.BLOCKED
    assert decision.reason_codes == ["no_usable_assessment_evidence"]
