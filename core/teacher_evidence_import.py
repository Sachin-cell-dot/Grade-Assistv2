"""Offline import of teacher-verified canonical answer-sheet evidence."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from pydantic import ValidationError

from core.local_result_cache import CachedGroqResult, LocalGroqResultCache
from core.models import VisionExtraction
from core.provenance import ExtractionProvenance

TEACHER_VERIFIED_SOURCE_TYPE = "teacher_verified_import"
TEACHER_VERIFIED_LABEL = "Teacher-verified local evidence"
TEACHER_VERIFIED_MODEL = "teacher-verified-local"
PRIOR_GROQ_TRANSCRIPT_SOURCE_TYPE = "prior_successful_groq_transcript"
PRIOR_GROQ_TRANSCRIPT_LABEL = "Recovered prior Groq evidence"


class TeacherEvidenceImportError(ValueError):
    pass


def import_teacher_verified_answer_evidence(
    json_path: str | Path,
    *,
    source_image_filename: str,
    note: str | None = None,
    policy_version: str,
    cache: LocalGroqResultCache | None = None,
) -> CachedGroqResult:
    """Validate canonical evidence and explicitly cache it without any provider."""
    source = Path(json_path)
    if not source.is_file() or source.suffix.lower() != ".json":
        raise TeacherEvidenceImportError("Teacher-verified evidence import requires an existing .json file.")
    if not source_image_filename or Path(source_image_filename).name != source_image_filename:
        raise TeacherEvidenceImportError("A source image filename (not a path) is required.")
    try:
        evidence = VisionExtraction.model_validate_json(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError, ValueError) as exc:
        raise TeacherEvidenceImportError("Teacher-verified evidence JSON does not match canonical VisionExtraction.") from exc
    provenance = ExtractionProvenance(
        provider=TEACHER_VERIFIED_SOURCE_TYPE,
        model=TEACHER_VERIFIED_MODEL,
        extracted_at=datetime.now(timezone.utc),
        source_image_identifier=source_image_filename,
        source_image_fingerprint_sha256=sha256(source.read_bytes()).hexdigest(),
        policy_version=policy_version,
        source_type=TEACHER_VERIFIED_SOURCE_TYPE,
        import_note=note or None,
    )
    return (cache or LocalGroqResultCache()).put(
        image_path=source,
        document_type="student_answer_sheet",
        model=TEACHER_VERIFIED_MODEL,
        policy_version=policy_version,
        result_payload=evidence.model_dump(mode="json"),
        provenance=provenance,
    )


def local_result_label(provenance: ExtractionProvenance) -> str:
    if provenance.source_type == TEACHER_VERIFIED_SOURCE_TYPE:
        return TEACHER_VERIFIED_LABEL
    if provenance.source_type == PRIOR_GROQ_TRANSCRIPT_SOURCE_TYPE:
        return PRIOR_GROQ_TRANSCRIPT_LABEL
    return "Verified local result"


def import_prior_successful_groq_transcript(
    json_path: str | Path,
    *,
    source_image_filename: str,
    model: str,
    policy_version: str,
    cache: LocalGroqResultCache | None = None,
) -> CachedGroqResult:
    """Cache canonical evidence recovered from an observed successful transcript.

    This is not live output and not teacher-verified evidence. It is always
    marked for teacher review.
    """
    source = Path(json_path)
    if not source.is_file() or source.suffix.lower() != ".json":
        raise TeacherEvidenceImportError("Recovered prior Groq evidence requires an existing .json file.")
    if not source_image_filename or Path(source_image_filename).name != source_image_filename:
        raise TeacherEvidenceImportError("A source image filename (not a path) is required.")
    try:
        evidence = VisionExtraction.model_validate_json(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError, ValueError) as exc:
        raise TeacherEvidenceImportError("Recovered prior Groq evidence JSON does not match canonical VisionExtraction.") from exc
    provenance = ExtractionProvenance(
        provider=PRIOR_GROQ_TRANSCRIPT_SOURCE_TYPE,
        model=model,
        extracted_at=datetime.now(timezone.utc),
        source_image_identifier=source_image_filename,
        source_image_fingerprint_sha256=sha256(source.read_bytes()).hexdigest(),
        policy_version=policy_version,
        source_type=PRIOR_GROQ_TRANSCRIPT_SOURCE_TYPE,
        requires_teacher_review=True,
    )
    return (cache or LocalGroqResultCache()).put(
        image_path=source,
        document_type="student_answer_sheet",
        model=model,
        policy_version=policy_version,
        result_payload=evidence.model_dump(mode="json"),
        provenance=provenance,
    )
