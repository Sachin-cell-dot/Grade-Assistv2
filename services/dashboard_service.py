"""Verified-only class dashboard data. No extraction or scoring decisions occur here."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import sqlite3

from core.models import VisionExtraction
from database.audit_store import LifecycleState
from tools.scoring import deterministic_total


MIN_TOPIC_OBSERVATIONS = 2


@dataclass(frozen=True)
class QuestionPerformance:
    question_id: str
    score: float | None
    maximum: float | None
    percentage: float | None
    status: str
    criteria: tuple[str, ...] = ()


@dataclass(frozen=True)
class StudentPerformance:
    audit_id: int
    student_name: str
    class_name: str | None
    parent_guardian_name: str | None
    parent_email: str | None
    assessment_name: str
    assessment_date: str | None
    marks: float | None
    maximum: float | None
    percentage: float | None
    review_status: str
    is_demo: bool = False
    demo_label: str | None = None
    questions: tuple[QuestionPerformance, ...] = ()


@dataclass(frozen=True)
class DashboardSummary:
    students: int
    assessments_processed: int
    average_score: float | None
    highest_score: float | None
    lowest_score: float | None
    needs_review: int


@dataclass(frozen=True)
class AssessmentActivityStep:
    label: str
    status: str
    detail: str


def _safe_percentage(score: float | None, maximum: float | None) -> float | None:
    if score is None or maximum is None or maximum <= 0:
        return None
    return round(score / maximum * 100, 2)


def _contacts(connection: sqlite3.Connection, name: str, class_name: str | None, is_demo: bool) -> tuple[str | None, str | None]:
    row = connection.execute(
        "SELECT parent_guardian_name, parent_email FROM students WHERE name = ? AND is_demo = ? AND (class_name = ? OR (? IS NULL AND class_name IS NULL)) ORDER BY id DESC LIMIT 1",
        (name, int(is_demo), class_name, class_name),
    ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def _suggestion_by_question(connection: sqlite3.Connection, audit_id: int) -> dict[str, dict]:
    rows = connection.execute(
        "SELECT payload_json FROM assessment_audit_events WHERE assessment_audit_id = ? AND event_type = 'SEMANTIC_SUGGESTION' ORDER BY id",
        (audit_id,),
    ).fetchall()
    return {payload.get("question_identifier"): payload for (raw,) in rows if (payload := json.loads(raw)).get("question_identifier")}


def verified_student_performance(connection: sqlite3.Connection) -> list[StudentPerformance]:
    """Read only finalized audit records; drafts/review-required records are excluded."""
    rows = connection.execute(
        "SELECT audit.id, audit.original_extraction_json, audit.created_at, audit.is_demo, audit.demo_label, "
        "student.name, student.class_name, assessment.assessment_name, assessment.assessment_date "
        "FROM assessment_audits AS audit "
        "LEFT JOIN assessments AS assessment ON assessment.audit_id = audit.id "
        "LEFT JOIN students AS student ON student.id = assessment.student_id "
        "WHERE audit.lifecycle_state = ? ORDER BY audit.created_at, audit.id",
        (LifecycleState.VERIFIED.value,),
    ).fetchall()
    result: list[StudentPerformance] = []
    for audit_id, raw, created_at, is_demo, demo_label, roster_name, roster_class, persisted_assessment_name, persisted_assessment_date in rows:
        extraction = VisionExtraction.model_validate_json(raw)
        name = roster_name or extraction.student.name or "Not available"
        class_name = roster_class if roster_name is not None else extraction.student.class_name
        parent_name, parent_email = _contacts(connection, name, class_name, bool(is_demo))
        reported = extraction.worksheet_reported_score
        marks = reported.obtained if reported else deterministic_total(extraction)
        maximum = reported.maximum if reported else None
        suggestions = _suggestion_by_question(connection, audit_id)
        questions: list[QuestionPerformance] = []
        for section in extraction.sections:
            for question in section.questions:
                score = question.teacher_marking.visible_individual_score.obtained if question.teacher_marking and question.teacher_marking.visible_individual_score else None
                suggestion = suggestions.get(question.identifier or "")
                question_maximum = suggestion.get("evaluation_maximum_marks") if suggestion else None
                criteria = tuple(item["criterion"]["text"] for item in suggestion.get("matched_rubric_criteria", []) if item.get("criterion", {}).get("text")) if suggestion else ()
                questions.append(QuestionPerformance(question.identifier or "Unlabelled", score, question_maximum, _safe_percentage(score, question_maximum), suggestion.get("status", "Not available") if suggestion else "Not available", criteria))
        result.append(StudentPerformance(
            audit_id=audit_id,
            student_name=name,
            class_name=class_name,
            parent_guardian_name=parent_name,
            parent_email=parent_email,
            assessment_name=persisted_assessment_name or extraction.student.subject or "Assessment",
            assessment_date=persisted_assessment_date or extraction.student.assessment_date or created_at,
            marks=marks,
            maximum=maximum,
            percentage=_safe_percentage(marks, maximum),
            review_status=LifecycleState.VERIFIED.value,
            is_demo=bool(is_demo),
            demo_label=demo_label,
            questions=tuple(questions),
        ))
    return result


def dashboard_summary(records: list[StudentPerformance]) -> DashboardSummary:
    scored = [record.marks for record in records if record.marks is not None]
    return DashboardSummary(
        students=len({(record.student_name, record.class_name) for record in records}),
        assessments_processed=len(records),
        average_score=round(sum(scored) / len(scored), 2) if scored else None,
        highest_score=max(scored) if scored else None,
        lowest_score=min(scored) if scored else None,
        needs_review=sum(record.review_status != LifecycleState.VERIFIED.value for record in records),
    )


def topic_insights(records: list[StudentPerformance], student_name: str) -> tuple[list[str], list[str]]:
    """Only label strengths/needs-attention after repeated available evidence."""
    values: dict[str, list[float]] = {}
    for record in records:
        if record.student_name != student_name:
            continue
        for question in record.questions:
            if question.percentage is not None:
                values.setdefault(question.question_id, []).append(question.percentage)
    averages = {topic: sum(scores) / len(scores) for topic, scores in values.items() if len(scores) >= MIN_TOPIC_OBSERVATIONS}
    if not averages:
        return [], []
    highest, lowest = max(averages.values()), min(averages.values())
    return ([topic for topic, value in averages.items() if value == highest], [topic for topic, value in averages.items() if value == lowest])


def assessment_activity(connection: sqlite3.Connection, audit_id: int, *, report_ready: bool) -> list[AssessmentActivityStep]:
    """Teacher-readable lifecycle view built only from persisted audit facts."""
    audit = connection.execute(
        "SELECT lifecycle_state FROM assessment_audits WHERE id = ?", (audit_id,)
    ).fetchone()
    if audit is None:
        return []
    lifecycle_state = audit[0]
    event_rows = connection.execute(
        "SELECT event_type, payload_json FROM assessment_audit_events WHERE assessment_audit_id = ? ORDER BY id", (audit_id,)
    ).fetchall()
    event_types = {event_type for event_type, _ in event_rows}
    route = None
    for event_type, payload_json in event_rows:
        if event_type == "ROUTING":
            route = json.loads(payload_json).get("route")
    email = connection.execute(
        "SELECT outcome FROM assessment_email_events WHERE assessment_audit_id = ? ORDER BY id DESC LIMIT 1", (audit_id,)
    ).fetchone()
    review_status = (
        "Completed" if lifecycle_state == LifecycleState.VERIFIED.value else "Required"
        if route == "TEACHER_REVIEW" else "Not required"
    )
    email_status = "Sent" if email and email[0] == "SENT" else "Send failed" if email and email[0] == "FAILED" else "Awaiting teacher approval"
    return [
        AssessmentActivityStep("Evidence extracted", "Recorded", "Immutable extraction evidence is stored."),
        AssessmentActivityStep("Evidence verified", "Completed" if "VERIFICATION" in event_types else "Not recorded", "Factual visible-mark checks only."),
        AssessmentActivityStep("Confidence/routing decision", route or "Not routed", "Routing is based on persisted verification evidence."),
        AssessmentActivityStep("Teacher review", review_status, "Teacher action is required before consequential evidence or mark decisions."),
        AssessmentActivityStep("Final VERIFIED result", lifecycle_state, "Only VERIFIED records feed class analytics."),
        AssessmentActivityStep("Dashboard updated", "Updated" if lifecycle_state == LifecycleState.VERIFIED.value else "Waiting", "Dashboard reads verified records only."),
        AssessmentActivityStep("Report ready", "Ready" if report_ready else "Waiting", "The current class report was prepared locally."),
        AssessmentActivityStep("Email", email_status, "Delivery occurs only after explicit teacher approval."),
    ]
