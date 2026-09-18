"""Deterministic routing of evidence verification outcomes; never grades work."""
from enum import Enum
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field

from core.evidence_verification import VerificationResult
from core.models import VisionExtraction, EvidenceModel

POLICY_VERSION = "teacher-review-routing-v1"


class ReviewRoute(str, Enum):
    AUTO_CONTINUE = "AUTO_CONTINUE"
    TEACHER_REVIEW = "TEACHER_REVIEW"
    BLOCKED = "BLOCKED"


class ReviewItem(EvidenceModel):
    id: str
    reason_code: str
    summary: str
    field_paths: list[str] = Field(default_factory=list)


def _review_item(reason_code: str, summary: str, field_paths: list[str] | None = None) -> ReviewItem:
    return ReviewItem(
        id=str(uuid5(NAMESPACE_URL, f"gradeassist:{reason_code}:{summary}:{field_paths or []}")),
        reason_code=reason_code,
        summary=summary,
        field_paths=field_paths or [],
    )


class RoutingDecision(EvidenceModel):
    route: ReviewRoute
    reason_codes: list[str] = Field(default_factory=list)
    teacher_summary: str
    review_items: list[ReviewItem] = Field(default_factory=list)
    policy_version: str = POLICY_VERSION


def _has_uncertain_handwritten_mark(extraction: VisionExtraction) -> bool:
    for section in extraction.sections:
        if section.visible_section_score and section.visible_section_score.uncertainty.is_uncertain:
            return True
        for question in section.questions:
            marking = question.teacher_marking
            if marking and (marking.uncertainty.is_uncertain or (marking.visible_individual_score and marking.visible_individual_score.uncertainty.is_uncertain)):
                return True
    return False


def route_extraction(extraction: VisionExtraction, verification: VerificationResult) -> RoutingDecision:
    """Route factual evidence only. No route ever asserts a correct answer or mark."""
    has_question_evidence = verification.evidence_summary.visible_questions > 0
    if not has_question_evidence:
        return RoutingDecision(
            route=ReviewRoute.BLOCKED,
            reason_codes=["no_usable_assessment_evidence"],
            teacher_summary="No usable question evidence was extracted; teacher review cannot begin from this result.",
            review_items=[_review_item("no_usable_assessment_evidence", "Extract a usable worksheet image or enter evidence manually.")],
        )

    items: list[ReviewItem] = []
    for check in verification.checks:
        if check.reason in {"individual_total_mismatch", "section_total_mismatch"}:
            paths = ["worksheet_reported_score.obtained"]
            if check.reason == "individual_total_mismatch":
                paths.extend(
                    f"sections[{section_index}].questions[{question_index}].teacher_marking.visible_individual_score.obtained"
                    for section_index, section in enumerate(extraction.sections)
                    for question_index, question in enumerate(section.questions)
                    if question.teacher_marking and question.teacher_marking.visible_individual_score is not None
                )
            items.append(_review_item(check.reason, check.explanation, paths))
    if _has_uncertain_handwritten_mark(extraction):
        items.append(_review_item("uncertain_handwritten_mark", "At least one handwritten mark is uncertain; confirm it visually before continuing."))

    if items:
        return RoutingDecision(
            route=ReviewRoute.TEACHER_REVIEW,
            reason_codes=[item.reason_code for item in items],
            teacher_summary=" ".join(item.summary for item in items),
            review_items=items,
        )
    if verification.status == "consistent":
        return RoutingDecision(route=ReviewRoute.AUTO_CONTINUE, teacher_summary="Visible evidence is complete and internally consistent.")
    return RoutingDecision(
        route=ReviewRoute.TEACHER_REVIEW,
        reason_codes=list(verification.blockers + verification.warnings),
        teacher_summary="Evidence is incomplete and requires teacher confirmation before continuing.",
        review_items=[_review_item(reason, "Review the incomplete extracted evidence.") for reason in verification.blockers + verification.warnings],
    )
