"""Verified-only class dashboard data. No extraction or scoring decisions occur here."""
from __future__ import annotations

from dataclasses import dataclass, field
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
        "SELECT id, original_extraction_json, created_at, is_demo, demo_label FROM assessment_audits WHERE lifecycle_state = ? ORDER BY created_at, id",
        (LifecycleState.VERIFIED.value,),
    ).fetchall()
    result: list[StudentPerformance] = []
    for audit_id, raw, created_at, is_demo, demo_label in rows:
        extraction = VisionExtraction.model_validate_json(raw)
        name = extraction.student.name or "Not available"
        class_name = extraction.student.class_name
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
            assessment_name=extraction.student.subject or "Assessment",
            assessment_date=extraction.student.assessment_date or created_at,
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
