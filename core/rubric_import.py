"""Local import for operator-provided verified live Groq document results.

No provider is called here. Files are typed, deterministically canonicalized,
and stored only in the ignored local cache with non-secret provenance.
"""
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from core.local_result_cache import CachedGroqResult, LocalGroqResultCache
from core.models import VisionExtraction
from core.provenance import build_provenance
from core.reference_models import QuestionPaperExtraction, RubricExtraction
from services.groq_vision_service import (
    GroqQuestionPaperExtraction,
    GroqRubricExtraction,
    GroqWorksheetExtraction,
    groq_dto_to_vision_extraction,
    groq_question_paper_to_extraction,
    groq_rubric_to_extraction,
)

VerifiedDocumentKind = Literal["question_paper", "rubric", "answer_sheet"]
IMPORTED_VERIFIED_LIVE_GROQ_PROVIDER = "imported_verified_live_groq_result"
_CACHE_DOCUMENT_TYPES = {"question_paper": "question_paper", "rubric": "rubric", "answer_sheet": "student_answer_sheet"}


class VerifiedDocumentImportError(ValueError):
    """Raised when a supplied operator document payload is not typed evidence."""


VerifiedRubricImportError = VerifiedDocumentImportError


def _canonicalize(raw_json: str, document_kind: VerifiedDocumentKind) -> BaseModel:
    try:
        if document_kind == "rubric":
            try:
                return RubricExtraction.model_validate_json(raw_json)
            except ValidationError:
                return groq_rubric_to_extraction(GroqRubricExtraction.model_validate_json(raw_json))
        if document_kind == "question_paper":
            try:
                return QuestionPaperExtraction.model_validate_json(raw_json)
            except ValidationError:
                return groq_question_paper_to_extraction(GroqQuestionPaperExtraction.model_validate_json(raw_json))
        try:
            return VisionExtraction.model_validate_json(raw_json)
        except ValidationError:
            return groq_dto_to_vision_extraction(GroqWorksheetExtraction.model_validate_json(raw_json))
    except (ValidationError, ValueError) as exc:
        raise VerifiedDocumentImportError(
            f"Verified {document_kind} JSON does not match its typed extraction model."
        ) from exc


def import_verified_live_document_json(
    json_path: str | Path,
    document_kind: VerifiedDocumentKind,
    *,
    model: str,
    policy_version: str,
    cache: LocalGroqResultCache | None = None,
) -> CachedGroqResult:
    """Validate and explicitly cache a verified live JSON result of one kind."""
    if document_kind not in _CACHE_DOCUMENT_TYPES:
        raise VerifiedDocumentImportError(f"Unsupported verified document kind: {document_kind}")
    source = Path(json_path)
    if not source.is_file():
        raise VerifiedDocumentImportError(f"Verified {document_kind} JSON file was not found: {source}")
    if source.suffix.lower() != ".json":
        raise VerifiedDocumentImportError("Verified document import accepts a .json file only.")
    try:
        canonical = _canonicalize(source.read_text(encoding="utf-8"), document_kind)
    except (OSError, UnicodeDecodeError) as exc:
        raise VerifiedDocumentImportError(f"Verified {document_kind} JSON could not be read.") from exc
    provenance = build_provenance(
        source,
        provider=IMPORTED_VERIFIED_LIVE_GROQ_PROVIDER,
        model=model,
        policy_version=policy_version,
    )
    return (cache or LocalGroqResultCache()).put(
        image_path=source,
        document_type=_CACHE_DOCUMENT_TYPES[document_kind],
        model=model,
        policy_version=policy_version,
        result_payload=canonical.model_dump(mode="json"),
        provenance=provenance,
    )


def import_verified_live_rubric_json(
    json_path: str | Path, *, model: str, policy_version: str, cache: LocalGroqResultCache | None = None,
) -> CachedGroqResult:
    """Compatibility wrapper for the original rubric-only import API."""
    return import_verified_live_document_json(
        json_path, "rubric", model=model, policy_version=policy_version, cache=cache
    )
