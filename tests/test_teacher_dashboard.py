from io import BytesIO

from openpyxl import load_workbook

from core.models import VisionExtraction
from core.partial_credit import PartialCreditSuggestion, SuggestionStatus
from core.teacher_review import ReviewAction
from database.audit_store import AuditStore, LifecycleState
from database.db import initialize_database
from services.assessment_finalization import finalize_verified_assessment, roster_contact, upsert_roster_contact
from services.dashboard_service import assessment_activity, dashboard_summary, verified_student_performance
import services.email_service as email_service
from services.email_service import email_send_readiness, report_email_preview, smtp_ready, student_report_body
from services.report_export import build_class_report


def _verified_record(tmp_path, name="Student", score=8, maximum=10):
    connection = initialize_database(tmp_path / "dashboard.db")
    store = AuditStore(connection)
    extraction = VisionExtraction.model_validate({
        "student": {"name": name, "class_name": "9", "subject": "English"},
        "worksheet_reported_score": {"obtained": score, "maximum": maximum},
        "sections": [{"questions": [{"identifier": "Q1", "student_answer": {"visible_text": "Answer"}, "teacher_marking": {"visible_individual_score": {"obtained": score}}}]}],
    })
    audit_id = store.create_extraction(extraction)
    store.record_suggestion(audit_id, PartialCreditSuggestion(question_identifier="Q1", teacher_awarded_mark=score, rubric_maximum_marks=maximum, evaluation_maximum_marks=maximum, suggested_mark=score, raw_semantic_coverage=score, rationale="Visible evidence.", status=SuggestionStatus.AGREES_WITH_TEACHER))
    store.transition(audit_id, LifecycleState.TEACHER_CONFIRMED, teacher_action=ReviewAction.CONFIRM_EXTRACTED)
    store.transition(audit_id, LifecycleState.VERIFIED, teacher_action=ReviewAction.CONFIRM_EXTRACTED)
    return connection


def test_dashboard_uses_verified_records_only_and_keeps_missing_parent_data_unavailable(tmp_path):
    connection = _verified_record(tmp_path)
    connection.execute("INSERT INTO assessment_audits(extraction_fingerprint, original_extraction_json, lifecycle_state) VALUES ('draft', '{\"sections\":[]}', 'REVIEW_REQUIRED')")
    connection.commit()
    records = verified_student_performance(connection)

    assert len(records) == 1
    assert records[0].student_name == "Student"
    assert records[0].parent_guardian_name is None
    assert records[0].parent_email is None
    assert records[0].percentage == 80
    assert records[0].questions[0].percentage == 80


def test_dashboard_summary_and_excel_report(tmp_path):
    connection = _verified_record(tmp_path, name="One", score=8)
    records = verified_student_performance(connection)
    summary = dashboard_summary(records)
    report = build_class_report(records, summary)
    workbook = load_workbook(BytesIO(report))

    assert summary.average_score == 8
    assert workbook.sheetnames == ["Student Summary", "Question-Topic Performance", "Class Summary"]
    assert workbook["Student Summary"][2][0].value == "One"
    assert workbook["Question-Topic Performance"][2][1].value == "Q1"


def test_email_preview_never_sends_and_smtp_requires_configuration():
    from config.settings import Settings

    preview = report_email_preview("GradeAssist class report", "Teacher summary", ["parent@example.com"])
    assert preview["To"] == "parent@example.com"
    assert "Teacher summary" in preview.get_content()
    assert smtp_ready(Settings(_env_file=None)) is False


def test_persisted_contact_is_used_and_missing_contact_is_never_fabricated(tmp_path):
    connection = _verified_record(tmp_path, name="Sachin")
    connection.execute("INSERT INTO students(name, class_name, parent_guardian_name, parent_email) VALUES (?, ?, ?, ?)", ("Sachin", "9", "Guardian", "guardian@example.com"))
    connection.commit()
    record = verified_student_performance(connection)[0]

    assert record.parent_guardian_name == "Guardian"
    assert record.parent_email == "guardian@example.com"
    body = student_report_body(student_name=record.student_name, assessment_name=record.assessment_name, marks=record.marks, maximum=record.maximum, percentage=record.percentage, verified_status=record.review_status, timestamp=record.assessment_date, strengths=[], needs_attention=[])
    assert "guardian@example.com" not in body
    assert "Strengths supported" not in body


def test_explicit_send_calls_smtp_while_preview_does_not(monkeypatch):
    from config.settings import Settings

    calls = []
    class FakeSMTP:
        def __init__(self, host, port, timeout): calls.append(("connect", host, port))
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def starttls(self): calls.append(("tls",))
        def login(self, username, password): calls.append(("login", username))
        def send_message(self, message): calls.append(("send", message["To"]))

    monkeypatch.setattr(email_service.smtplib, "SMTP", FakeSMTP)
    settings = Settings(_env_file=None, smtp_host="smtp.example.com", smtp_from_email="teacher@example.com")
    report_email_preview("Preview", "Body", ["parent@example.com"])
    assert calls == []
    email_service.send_report_email(settings, ["parent@example.com"], "Report", "Body", b"report")
    assert ("send", "parent@example.com") in calls


def test_activity_is_built_from_persisted_lifecycle_and_email_audit(tmp_path):
    connection = _verified_record(tmp_path, name="Activity student")
    record = verified_student_performance(connection)[0]
    AuditStore(connection).record_email_event(record.audit_id, "ATTEMPTED", "Teacher explicitly selected Send Email.")
    AuditStore(connection).record_email_event(record.audit_id, "SENT", "SMTP accepted the teacher-approved report for delivery.")

    activity = assessment_activity(connection, record.audit_id, report_ready=True)
    statuses = {item.label: item.status for item in activity}
    assert statuses["Evidence extracted"] == "Recorded"
    assert statuses["Evidence verified"] == "Not recorded"
    assert statuses["Final VERIFIED result"] == "VERIFIED"
    assert statuses["Dashboard updated"] == "Updated"
    assert statuses["Report ready"] == "Ready"
    assert statuses["Email"] == "Sent"


def test_email_readiness_requires_teacher_selection_email_and_smtp_configuration():
    from config.settings import Settings

    settings = Settings(_env_file=None)
    assert email_send_readiness(settings, selected_count=0, recipient_count=0)[0] is False
    assert "Select at least one" in email_send_readiness(settings, selected_count=0, recipient_count=0)[1]
    assert "Email not available" in email_send_readiness(settings, selected_count=1, recipient_count=0)[1]
    assert "SMTP configuration required" in email_send_readiness(settings, selected_count=1, recipient_count=1)[1]
    configured = Settings(_env_file=None, smtp_host="smtp.example.com", smtp_from_email="teacher@example.com")
    assert email_send_readiness(configured, selected_count=1, recipient_count=1) == (
        True, "Ready for teacher approval. Sending occurs only after you click Send Email."
    )


def test_finalized_real_assessment_flows_to_verified_dashboard_with_persisted_contact(tmp_path):
    connection = initialize_database(tmp_path / "pipeline.db")
    store = AuditStore(connection)
    extraction = VisionExtraction.model_validate({
        "student": {"name": "Sanjay Kumar M", "class_name": "9", "subject": "English", "assessment_date": "19/09/2026"},
        "worksheet_reported_score": {"obtained": 10, "maximum": 20},
        "sections": [{"questions": [{"identifier": "Q1", "teacher_marking": {"visible_individual_score": {"obtained": 10}}}]}],
    })
    audit_id = store.create_extraction(extraction)
    store.transition(audit_id, LifecycleState.TEACHER_CONFIRMED, teacher_action=ReviewAction.CONFIRM_EXTRACTED)

    finalize_verified_assessment(
        connection, audit_id, parent_guardian_name="Guardian", parent_email="guardian@example.com"
    )
    records = verified_student_performance(connection)
    assert len(records) == 1
    assert records[0].student_name == "Sanjay Kumar M"
    assert records[0].parent_guardian_name == "Guardian"
    assert records[0].parent_email == "guardian@example.com"
    assert records[0].marks == 10
    assert records[0].maximum == 20
    assert records[0].percentage == 50
    assert records[0].assessment_date == "19/09/2026"
    assert dashboard_summary(records).average_score == 10
    assert connection.execute("SELECT verified_status, worksheet_reported_score, maximum_marks FROM assessments WHERE audit_id = ?", (audit_id,)).fetchone() == ("VERIFIED", 10.0, 20.0)
    report = build_class_report(records, dashboard_summary(records))
    assert load_workbook(BytesIO(report))["Student Summary"][2][0].value == "Sanjay Kumar M"


def test_review_required_assessment_cannot_be_finalized_or_leak_into_dashboard(tmp_path):
    connection = initialize_database(tmp_path / "pipeline.db")
    store = AuditStore(connection)
    audit_id = store.create_extraction(VisionExtraction.model_validate({"student": {"name": "Review student", "class_name": "9"}}))
    store.transition(audit_id, LifecycleState.REVIEW_REQUIRED, rationale="Visible score mismatch.")

    import pytest
    with pytest.raises(ValueError, match="teacher-confirmed"):
        finalize_verified_assessment(connection, audit_id)
    assert verified_student_performance(connection) == []


def test_roster_upsert_does_not_modify_demo_or_fabricate_missing_contact(tmp_path):
    connection = initialize_database(tmp_path / "roster.db")
    connection.execute("INSERT INTO students(name, class_name, parent_email, is_demo) VALUES ('Sanjay Kumar M', '9', 'demo@example.com', 1)")
    connection.commit()
    student_id = upsert_roster_contact(
        connection, student_name="Sanjay Kumar M", class_name="9", roll_number=None,
        parent_guardian_name=None, parent_email=None,
    )
    assert student_id is not None
    assert roster_contact(connection, student_name="Sanjay Kumar M", class_name="9") == (None, None)
    assert connection.execute("SELECT parent_email FROM students WHERE is_demo = 1").fetchone()[0] == "demo@example.com"


def test_teacher_confirmed_roster_identity_is_used_by_dashboard_without_mutating_extraction(tmp_path):
    connection = initialize_database(tmp_path / "identity.db")
    store = AuditStore(connection)
    extraction = VisionExtraction.model_validate({
        "student": {"name": "Extracted spelling", "class_name": "9", "subject": "English"},
        "worksheet_reported_score": {"obtained": 10, "maximum": 20},
        "sections": [{"questions": [{"identifier": "Q1", "teacher_marking": {"visible_individual_score": {"obtained": 10}}}]}],
    })
    audit_id = store.create_extraction(extraction)
    store.transition(audit_id, LifecycleState.TEACHER_CONFIRMED, teacher_action=ReviewAction.CONFIRM_EXTRACTED)
    finalize_verified_assessment(
        connection, audit_id, roster_student_name="Sanjay Kumar M", roster_class_name="9"
    )
    assert store.original_extraction(audit_id).student.name == "Extracted spelling"
    assert verified_student_performance(connection)[0].student_name == "Sanjay Kumar M"
