"""Seed explicitly labelled, local-only demo records from existing evidence.

This command makes no provider or model calls.  It creates distinct demo audit
rows and never promotes, rewrites, or deletes real assessment records.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import sqlite3

from config.settings import Settings, get_settings
from core.local_result_cache import LocalGroqResultCache
from core.models import VisionExtraction
from core.provenance import ExtractionProvenance
from core.teacher_review import ReviewAction
from database.audit_store import AuditStore, LifecycleState
from database.db import initialize_database
from ui.extraction_demo import POLICY_VERSION


DEMO_LABEL = "LOCAL DEMO DATA"
_PROCESSED_DIRECTORY = Path("data/processed")


def _canonical_file(path: Path) -> VisionExtraction:
    return VisionExtraction.model_validate_json(path.read_text(encoding="utf-8"))


def _existing_demo_sources(cache: LocalGroqResultCache | None = None) -> list[tuple[str, VisionExtraction, str, str]]:
    """Return only already present, validated sources; never extract or infer."""
    sachin_path = _PROCESSED_DIRECTORY / "sachin_prior_groq_evidence.json"
    if not sachin_path.is_file():
        raise FileNotFoundError(f"Required existing demo evidence is missing: {sachin_path}")
    sachin = _canonical_file(sachin_path)

    cache = cache or LocalGroqResultCache()
    rufina_cached = cache.for_source_image_identifier(
        "student_answer_sheet", "Rufina_Answersheet.jpeg", POLICY_VERSION
    )
    if rufina_cached is not None and rufina_cached.provenance.source_type == "teacher_verified_import":
        rufina = VisionExtraction.model_validate(rufina_cached.result_payload)
        rufina_source = rufina_cached.provenance.source_image_identifier
    else:
        rufina_path = _PROCESSED_DIRECTORY / "rufina_manual_review_draft.json"
        if not rufina_path.is_file():
            raise FileNotFoundError("Required existing Rufina demo evidence is missing from the verified cache and data/processed.")
        rufina = _canonical_file(rufina_path)
        rufina_source = "Rufina_Answersheet.jpeg"
    return [("Sachin", sachin, "Sachin_Answersheet.jpeg", str(sachin_path)), ("Rufina", rufina, rufina_source, "verified local cache")]


def _configured_demo_email(student_name: str | None, settings: Settings) -> str | None:
    normalized = (student_name or "").casefold()
    if normalized.startswith("sachin"):
        return settings.demo_sachin_email or None
    if normalized.startswith("rufina"):
        return settings.demo_rufina_email or None
    return None


def _upsert_demo_student(connection: sqlite3.Connection, extraction: VisionExtraction, email: str | None) -> None:
    name = extraction.student.name or "Not available"
    class_name = extraction.student.class_name
    row = connection.execute(
        "SELECT id FROM students WHERE name = ? AND (class_name = ? OR (? IS NULL AND class_name IS NULL)) AND is_demo = 1 ORDER BY id DESC LIMIT 1",
        (name, class_name, class_name),
    ).fetchone()
    if row is None:
        connection.execute(
            "INSERT INTO students(name, class_name, parent_email, is_demo) VALUES (?, ?, ?, 1)",
            (name, class_name, email),
        )
    else:
        # This row is already explicitly local demo data; never update a real row.
        connection.execute("UPDATE students SET parent_email = ? WHERE id = ? AND is_demo = 1", (email, row[0]))
    connection.commit()


def _demo_provenance(extraction: VisionExtraction, source_image: str, source_description: str) -> ExtractionProvenance:
    return ExtractionProvenance(
        provider="local_demo_seed",
        model="existing-validated-evidence",
        extracted_at=datetime.now(timezone.utc),
        source_image_identifier=source_image,
        source_image_fingerprint_sha256=sha256(extraction.model_dump_json().encode("utf-8")).hexdigest(),
        policy_version=POLICY_VERSION,
        source_type="local_demo_seed",
        import_note=f"{DEMO_LABEL}; seeded from {source_description} without extraction or scoring.",
        requires_teacher_review=False,
    )


def seed_demo_records(
    connection: sqlite3.Connection,
    *,
    settings: Settings | None = None,
    sources: list[tuple[str, VisionExtraction, str, str]] | None = None,
) -> list[int]:
    """Create/reuse separate demo audits, then explicitly confirm the demo rows."""
    settings = settings or get_settings()
    store = AuditStore(connection)
    audit_ids: list[int] = []
    for _, extraction, image_name, source_description in sources or _existing_demo_sources():
        _upsert_demo_student(connection, extraction, _configured_demo_email(extraction.student.name, settings))
        audit_id = store.create_demo_extraction(extraction, _demo_provenance(extraction, image_name, source_description))
        if store.lifecycle_state(audit_id) == LifecycleState.EXTRACTED:
            rationale = "LOCAL DEMO DATA: explicit operator --seed-demo confirmation; source evidence was not altered."
            store.transition(audit_id, LifecycleState.TEACHER_CONFIRMED, teacher_action=ReviewAction.CONFIRM_EXTRACTED, rationale=rationale)
            store.transition(audit_id, LifecycleState.VERIFIED, teacher_action=ReviewAction.CONFIRM_EXTRACTED, rationale=rationale)
        audit_ids.append(audit_id)
    return audit_ids


def clear_demo_records(connection: sqlite3.Connection) -> int:
    """Remove only rows explicitly marked as local demo data."""
    demo_ids = [row[0] for row in connection.execute("SELECT id FROM assessment_audits WHERE is_demo = 1")]
    if demo_ids:
        placeholders = ",".join("?" for _ in demo_ids)
        connection.execute(f"DELETE FROM assessment_audit_events WHERE assessment_audit_id IN ({placeholders})", demo_ids)
        connection.execute(f"DELETE FROM assessment_audits WHERE id IN ({placeholders})", demo_ids)
    connection.execute("DELETE FROM assessments WHERE is_demo = 1")
    # A future non-demo assessment must protect its referenced student even if
    # that student was originally introduced by a demo seed.
    connection.execute(
        "DELETE FROM students WHERE is_demo = 1 "
        "AND NOT EXISTS (SELECT 1 FROM assessments WHERE assessments.student_id = students.id AND assessments.is_demo = 0)"
    )
    connection.commit()
    return len(demo_ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed or clear explicit GradeAssist local demo records.")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--seed-demo", action="store_true", help="Create verified LOCAL DEMO DATA records from existing local evidence.")
    actions.add_argument("--clear-demo", action="store_true", help="Remove only LOCAL DEMO DATA records.")
    args = parser.parse_args(argv)
    connection = initialize_database()
    try:
        if args.seed_demo:
            audit_ids = seed_demo_records(connection)
            print(f"{DEMO_LABEL}: seeded {len(audit_ids)} verified demo audit record(s): {', '.join(map(str, audit_ids))}")
        else:
            print(f"{DEMO_LABEL}: removed {clear_demo_records(connection)} demo audit record(s).")
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
