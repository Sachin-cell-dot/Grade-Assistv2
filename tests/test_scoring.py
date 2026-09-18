from core.models import VisionExtraction
from tools.scoring import deterministic_total

def test_total_uses_individual_marks_not_reported_total():
    extraction = VisionExtraction.model_validate({
        "worksheet_reported_score": {"obtained": 99},
        "sections": [{"identifier": "A", "questions": [
            {"identifier": "1", "teacher_marking": {"visible_individual_score": {"obtained": 2}}},
            {"identifier": "2", "teacher_marking": {"visible_individual_score": {"obtained": 1.5}}},
        ]}],
    })
    assert deterministic_total(extraction) == 3.5


def test_total_is_unavailable_without_visible_individual_marks():
    assert deterministic_total(VisionExtraction.model_validate({"sections": [{"identifier": "A", "questions": [{"identifier": "1"}]}]})) is None


def test_total_can_be_zero_when_visible_zero_mark_exists():
    extraction = VisionExtraction.model_validate({"sections": [{"identifier": "A", "questions": [{"identifier": "1", "teacher_marking": {"visible_individual_score": {"obtained": 0}}}]}]})
    assert deterministic_total(extraction) == 0.0
