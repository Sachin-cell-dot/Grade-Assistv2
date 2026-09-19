from config.settings import Settings
from core.models import VisionExtraction
from core.teacher_review import ReviewAction
from database.audit_store import AuditStore, LifecycleState
from database.db import initialize_database
from services.dashboard_service import verified_student_performance
from tools.seed_demo_records import DEMO_LABEL, clear_demo_records, seed_demo_records


def _evidence(name: str, score: float) -> VisionExtraction:
    return VisionExtraction.model_validate({
        "student": {"name": name, "class_name": "9", "subject": "English"},
        "worksheet_reported_score": {"obtained": score, "maximum": 20},
        "sections": [{"questions": [{"identifier": "Q1", "student_answer": {"visible_text": "Visible answer"}, "teacher_marking": {"visible_individual_score": {"obtained": score}}}]}],
    })


def _sources():
    return [
        ("Sachin", _evidence("Sachin", 10), "Sachin_Answersheet.jpeg", "test evidence"),
        ("Rufina", _evidence("Rufina", 11), "Rufina_Answersheet.jpeg", "test evidence"),
    ]


def test_demo_seed_is_deterministic_verified_and_visible_to_normal_dashboard(tmp_path):
    connection = initialize_database(tmp_path / "demo.db")
    settings = Settings(_env_file=None, demo_sachin_email="sachin-demo@example.com", demo_rufina_email="rufina-demo@example.com")

    ids = seed_demo_records(connection, settings=settings, sources=_sources())
    assert len(ids) == 2
    assert seed_demo_records(connection, settings=settings, sources=_sources()) == ids
    audits = connection.execute("SELECT lifecycle_state, is_demo, demo_label FROM assessment_audits ORDER BY id").fetchall()
    assert audits == [(LifecycleState.VERIFIED.value, 1, DEMO_LABEL), (LifecycleState.VERIFIED.value, 1, DEMO_LABEL)]
    students = connection.execute("SELECT name, parent_email, is_demo FROM students ORDER BY name").fetchall()
    assert students == [("Rufina", "rufina-demo@example.com", 1), ("Sachin", "sachin-demo@example.com", 1)]
    records = verified_student_performance(connection)
    assert {record.student_name for record in records} == {"Sachin", "Rufina"}
    assert all(record.is_demo and record.demo_label == DEMO_LABEL for record in records)


def test_demo_seed_keeps_unconfigured_email_unavailable_and_clear_only_removes_demo(tmp_path):
    connection = initialize_database(tmp_path / "demo.db")
    real = _evidence("Real student", 8)
    real_id = AuditStore(connection).create_extraction(real)
    AuditStore(connection).transition(real_id, LifecycleState.TEACHER_CONFIRMED, teacher_action=ReviewAction.CONFIRM_EXTRACTED)
    AuditStore(connection).transition(real_id, LifecycleState.VERIFIED, teacher_action=ReviewAction.CONFIRM_EXTRACTED)
    connection.execute("INSERT INTO students(name, class_name, parent_email, is_demo) VALUES ('Real student', '9', 'real@example.com', 0)")
    connection.commit()

    seed_demo_records(connection, settings=Settings(_env_file=None), sources=_sources())
    demo_emails = connection.execute("SELECT parent_email FROM students WHERE is_demo = 1").fetchall()
    assert demo_emails == [(None,), (None,)]
    assert clear_demo_records(connection) == 2
    assert connection.execute("SELECT count(*) FROM assessment_audits WHERE is_demo = 0").fetchone()[0] == 1
    assert connection.execute("SELECT parent_email FROM students WHERE is_demo = 0").fetchone()[0] == "real@example.com"
