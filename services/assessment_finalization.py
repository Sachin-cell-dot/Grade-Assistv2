"""Teacher-triggered persistence of a verified assessment and its roster contact."""
from __future__ import annotations

import sqlite3

from core.teacher_review import ReviewAction
from database.audit_store import AuditStore, LifecycleState
from tools.scoring import deterministic_total


def roster_contact(
    connection: sqlite3.Connection, *, student_name: str | None, class_name: str | None
) -> tuple[str | None, str | None]:
    """Read only matching non-demo roster contact data; never fabricate it."""
    if not student_name:
        return None, None
    row = connection.execute(
        "SELECT parent_guardian_name, parent_email FROM students "
        "WHERE name = ? AND is_demo = 0 AND (class_name = ? OR (? IS NULL AND class_name IS NULL)) "
        "ORDER BY id DESC LIMIT 1",
        (student_name, class_name, class_name),
    ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def upsert_roster_contact(
    connection: sqlite3.Connection,
    *,
    student_name: str | None,
    class_name: str | None,
    roll_number: str | None,
    parent_guardian_name: str | None,
    parent_email: str | None,
) -> int | None:
    """Persist a teacher-supplied contact only for the real (non-demo) roster."""
    if not student_name:
        return None
    row = connection.execute(
        "SELECT id FROM students WHERE name = ? AND is_demo = 0 "
        "AND (class_name = ? OR (? IS NULL AND class_name IS NULL)) ORDER BY id DESC LIMIT 1",
        (student_name, class_name, class_name),
    ).fetchone()
    if row is None:
        cursor = connection.execute(
            "INSERT INTO students(name, roll_number, class_name, parent_guardian_name, parent_email, is_demo) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            (student_name, roll_number or None, class_name, parent_guardian_name or None, parent_email or None),
        )
        student_id = int(cursor.lastrowid)
    else:
        student_id = int(row[0])
        connection.execute(
            "UPDATE students SET roll_number = COALESCE(?, roll_number), parent_guardian_name = ?, parent_email = ? "
            "WHERE id = ? AND is_demo = 0",
            (roll_number or None, parent_guardian_name or None, parent_email or None, student_id),
        )
    connection.commit()
    return student_id


def finalize_verified_assessment(
    connection: sqlite3.Connection,
    audit_id: int,
    *,
    roster_student_name: str | None = None,
    roster_class_name: str | None = None,
    parent_guardian_name: str | None = None,
    parent_email: str | None = None,
) -> int:
    """Complete the existing teacher-confirmed audit lifecycle and persist its link.

    This is deliberately unavailable for REVIEW_REQUIRED or other unconfirmed
    audits. It uses extracted visible marks only and does not recalculate or
    modify evidence.
    """
    store = AuditStore(connection)
    if store.lifecycle_state(audit_id) != LifecycleState.TEACHER_CONFIRMED:
        raise ValueError("Only a teacher-confirmed assessment can be finalized as VERIFIED.")
    extraction = store.original_extraction(audit_id)
    student_name = roster_student_name or extraction.student.name
    class_name = roster_class_name or extraction.student.class_name
    student_id = upsert_roster_contact(
        connection,
        student_name=student_name,
        class_name=class_name,
        roll_number=extraction.student.roll_number,
        parent_guardian_name=parent_guardian_name,
        parent_email=parent_email,
    )
    reported = extraction.worksheet_reported_score
    marks = reported.obtained if reported else deterministic_total(extraction)
    maximum = reported.maximum if reported else None
    connection.execute(
        "INSERT INTO assessments(student_id, audit_id, assessment_name, subject, assessment_date, worksheet_reported_score, maximum_marks, verified_status, verified_at, is_demo) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 0) "
        "ON CONFLICT(audit_id) DO UPDATE SET student_id = excluded.student_id, assessment_name = excluded.assessment_name, "
        "subject = excluded.subject, assessment_date = excluded.assessment_date, worksheet_reported_score = excluded.worksheet_reported_score, "
        "maximum_marks = excluded.maximum_marks, verified_status = excluded.verified_status, verified_at = CURRENT_TIMESTAMP",
        (student_id, audit_id, extraction.student.subject or "Assessment", extraction.student.subject, extraction.student.assessment_date,
         marks, maximum, LifecycleState.VERIFIED.value),
    )
    connection.commit()
    store.transition(
        audit_id, LifecycleState.VERIFIED, teacher_action=ReviewAction.CONFIRM_EXTRACTED,
        rationale="Teacher explicitly finalized the confirmed assessment for dashboard reporting.",
    )
    return audit_id
