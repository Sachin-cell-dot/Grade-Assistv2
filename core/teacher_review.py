"""Append-only teacher evidence corrections and deterministic overlays."""
from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
import re

from pydantic import Field, model_validator

from core.evidence_verification import VerificationResult, verify_extraction
from core.models import EvidenceModel, VisionExtraction
from core.routing import RoutingDecision, route_extraction

REVIEW_POLICY_VERSION = "teacher-evidence-overlay-v1"
_MARK_PATH = re.compile(r"^sections\[(\d+)\]\.questions\[(\d+)\]\.teacher_marking\.visible_individual_score\.obtained$")
_TOTAL_PATH = "worksheet_reported_score.obtained"


class ReviewAction(str, Enum):
    CONFIRM_EXTRACTED = "CONFIRM_EXTRACTED"
    CORRECT_EVIDENCE = "CORRECT_EVIDENCE"
    DEFER = "DEFER"


class TeacherCorrection(EvidenceModel):
    review_item_id: str
    field_path: str
    original_value: float
    corrected_value: float = Field(ge=0)
    teacher_note: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def only_supported_paths(self):
        if self.field_path != _TOTAL_PATH and not _MARK_PATH.fullmatch(self.field_path):
            raise ValueError("Correction field path is not supported.")
        return self


class ReviewResolution(EvidenceModel):
    action: ReviewAction
    corrections: list[TeacherCorrection] = Field(default_factory=list)
    effective_extraction: VisionExtraction
    verification: VerificationResult | None = None
    routing: RoutingDecision | None = None
    policy_version: str = REVIEW_POLICY_VERSION


def _field_value(payload: dict, path: str) -> float:
    if path == _TOTAL_PATH:
        score = payload.get("worksheet_reported_score")
        if not isinstance(score, dict) or not isinstance(score.get("obtained"), (int, float)):
            raise ValueError("Worksheet reported total is not a visible numeric value.")
        return float(score["obtained"])
    match = _MARK_PATH.fullmatch(path)
    if not match:
        raise ValueError("Correction field path is not supported.")
    section_index, question_index = (int(part) for part in match.groups())
    try:
        score = payload["sections"][section_index]["questions"][question_index]["teacher_marking"]["visible_individual_score"]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError("Correction field path does not point to a visible individual mark.") from exc
    if not isinstance(score, dict) or not isinstance(score.get("obtained"), (int, float)):
        raise ValueError("Correction field path does not point to a visible individual mark.")
    return float(score["obtained"])


def correction_value(extraction: VisionExtraction, field_path: str) -> float:
    """Read a permitted value without mutating original evidence."""
    return _field_value(extraction.model_dump(mode="json"), field_path)


def apply_correction_overlay(original: VisionExtraction, corrections: list[TeacherCorrection]) -> VisionExtraction:
    """Apply corrections to a deep copy; original extraction remains immutable."""
    payload = deepcopy(original.model_dump(mode="json"))
    for correction in corrections:
        actual = _field_value(payload, correction.field_path)
        if abs(actual - correction.original_value) > 1e-9:
            raise ValueError("Correction original value does not match the current effective evidence.")
        if correction.field_path == _TOTAL_PATH:
            payload["worksheet_reported_score"]["obtained"] = correction.corrected_value
        else:
            section_index, question_index = (int(part) for part in _MARK_PATH.fullmatch(correction.field_path).groups())
            payload["sections"][section_index]["questions"][question_index]["teacher_marking"]["visible_individual_score"]["obtained"] = correction.corrected_value
    return VisionExtraction.model_validate(payload)


def resolve_teacher_review(original: VisionExtraction, action: ReviewAction, corrections: list[TeacherCorrection] | None = None) -> ReviewResolution:
    """Execute only an explicit teacher action; DEFER intentionally does no verification."""
    corrections = list(corrections or [])
    if action == ReviewAction.DEFER:
        return ReviewResolution(action=action, corrections=corrections, effective_extraction=original.model_copy(deep=True))
    if action == ReviewAction.CONFIRM_EXTRACTED and corrections:
        raise ValueError("CONFIRM_EXTRACTED cannot include corrections.")
    if action == ReviewAction.CORRECT_EVIDENCE and not corrections:
        raise ValueError("CORRECT_EVIDENCE requires at least one correction.")
    effective = apply_correction_overlay(original, corrections)
    verification = verify_extraction(effective)
    return ReviewResolution(action=action, corrections=corrections, effective_extraction=effective, verification=verification, routing=route_extraction(effective, verification))
