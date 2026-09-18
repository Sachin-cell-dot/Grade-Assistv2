import sqlite3

import pytest

from core.evidence_verification import verify_extraction
from core.models import VisionExtraction
from core.partial_credit import (
    PartialCreditSuggestion,
    SuggestionStatus,
    TeacherDispositionAction,
    record_teacher_disposition,
)
from core.routing import route_extraction
from core.teacher_review import ReviewAction, TeacherCorrection
from database.audit_store import AuditStore, LifecycleState
from database.db import initialize_database


def extraction():
    return VisionExtraction.model_validate({"sections": [{"questions": [{"identifier": "1", "student_answer": {"visible_text": "Answer"}, "teacher_marking": {"visible_individual_score": {"obtained": 1}}}]}]})


def store(tmp_path):
    return AuditStore(initialize_database(tmp_path / "audit.db"))


def test_persists_reloads_and_keeps_original_extraction_immutable(tmp_path):
    audit = store(tmp_path)
    original = extraction()
    audit_id = audit.create_extraction(original)
    audit.record_correction(audit_id, TeacherCorrection(review_item_id="item", field_path="sections[0].questions[0].teacher_marking.visible_individual_score.obtained", original_value=1, corrected_value=2))
    reloaded = AuditStore(initialize_database(tmp_path / "audit.db")).original_extraction(audit_id)
    assert reloaded == original
    assert reloaded.sections[0].questions[0].teacher_marking.visible_individual_score.obtained == 1


def test_append_only_history_and_route_state(tmp_path):
    audit = store(tmp_path)
    audit_id = audit.create_extraction(extraction())
    verification = verify_extraction(extraction())
    audit.record_verification(audit_id, verification)
    audit.apply_route(audit_id, route_extraction(extraction(), verification))
    rows = audit.connection.execute("SELECT event_type FROM assessment_audit_events WHERE assessment_audit_id = ?", (audit_id,)).fetchall()
    assert [row[0] for row in rows] == ["VERIFICATION", "ROUTING", "LIFECYCLE"]
    assert audit.lifecycle_state(audit_id) == LifecycleState.REVIEW_REQUIRED


def test_valid_transitions_require_explicit_teacher_confirmation(tmp_path):
    audit = store(tmp_path)
    audit_id = audit.create_extraction(extraction())
    with pytest.raises(ValueError):
        audit.transition(audit_id, LifecycleState.VERIFIED)
    audit.transition(audit_id, LifecycleState.TEACHER_CONFIRMED, teacher_action=ReviewAction.CONFIRM_EXTRACTED)
    audit.transition(audit_id, LifecycleState.VERIFIED, teacher_action=ReviewAction.CONFIRM_EXTRACTED)
    assert audit.lifecycle_state(audit_id) == LifecycleState.VERIFIED


def test_deferred_record_never_becomes_verified(tmp_path):
    audit = store(tmp_path)
    audit_id = audit.create_extraction(extraction())
    audit.transition(audit_id, LifecycleState.DEFERRED, rationale="Teacher deferred review.")
    with pytest.raises(ValueError):
        audit.transition(audit_id, LifecycleState.VERIFIED, teacher_action=ReviewAction.CONFIRM_EXTRACTED)
    assert audit.lifecycle_state(audit_id) == LifecycleState.DEFERRED


def test_suggestions_and_teacher_decisions_are_append_only_and_counted(tmp_path):
    audit = store(tmp_path)
    audit_id = audit.create_extraction(extraction())
    suggestion = PartialCreditSuggestion(question_identifier="1", teacher_awarded_mark=1, rubric_maximum_marks=2, suggested_mark=2, rationale="Teacher review required.", status=SuggestionStatus.SUGGEST_REVIEW)
    audit.record_suggestion(audit_id, suggestion)
    audit.record_partial_credit_decision(audit_id, record_teacher_disposition(suggestion, TeacherDispositionAction.ACCEPT_SUGGESTION))
    audit.record_partial_credit_decision(audit_id, record_teacher_disposition(suggestion, TeacherDispositionAction.EDIT_FINAL_MARK, edited_final_mark=1.5))
    audit.record_partial_credit_decision(audit_id, record_teacher_disposition(suggestion, TeacherDispositionAction.PROCEED_WITH_TEACHER_MARK))
    summary = audit.summary(audit_id)
    assert (summary.accepted_count, summary.edited_count, summary.kept_teacher_mark_count) == (1, 1, 1)
    assert summary.suggestion == 2
