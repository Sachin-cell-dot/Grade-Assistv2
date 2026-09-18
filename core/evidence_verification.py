"""Pure deterministic checks over canonical worksheet evidence."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.models import Question, VisionExtraction

POLICY_VERSION = "evidence-verification-v1"


class EvidenceCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    status: Literal["pass", "warning", "fail"]
    reason: str | None = None
    explanation: str


class EvidenceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    visible_sections: int
    sections_with_score: int
    section_score_coverage: float | None
    visible_section_score_sum: float | None
    worksheet_reported_obtained: float | None
    worksheet_reported_maximum: float | None
    visible_questions: int
    questions_with_text: int
    questions_with_student_answer: int
    questions_with_teacher_symbol: int
    questions_with_individual_mark: int
    identity_fields_present: list[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["consistent", "needs_review", "insufficient_evidence"]
    checks: list[EvidenceCheck]
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence_summary: EvidenceSummary
    policy_version: str = POLICY_VERSION


def _all_questions(extraction: VisionExtraction) -> list[Question]:
    questions: list[Question] = []
    for section in extraction.sections:
        questions.extend(section.questions)
        for question in section.questions:
            questions.extend(
                Question(
                    identifier=subquestion.identifier,
                    student_answer=subquestion.student_answer,
                    teacher_marking=subquestion.teacher_marking,
                )
                for subquestion in question.subquestions
            )
    return questions


def _display_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def verify_extraction(extraction: VisionExtraction) -> VerificationResult:
    """Return factual evidence consistency; never changes extraction evidence."""
    sections = extraction.sections
    questions = _all_questions(extraction)
    section_scores = [section.visible_section_score.obtained for section in sections if section.visible_section_score is not None]
    reported = extraction.worksheet_reported_score
    identity = {
        "name": extraction.student.name,
        "roll_number": extraction.student.roll_number,
        "class_name": extraction.student.class_name,
        "subject": extraction.student.subject,
        "assessment_date": extraction.student.assessment_date,
    }
    summary = EvidenceSummary(
        visible_sections=len(sections),
        sections_with_score=len(section_scores),
        section_score_coverage=(len(section_scores) / len(sections)) if sections else None,
        visible_section_score_sum=sum(section_scores) if sections and len(section_scores) == len(sections) else None,
        worksheet_reported_obtained=reported.obtained if reported else None,
        worksheet_reported_maximum=reported.maximum if reported else None,
        visible_questions=len(questions),
        questions_with_text=sum(question.question_text is not None for question in questions),
        questions_with_student_answer=sum(question.student_answer.visible_text is not None for question in questions),
        questions_with_teacher_symbol=sum(bool(question.teacher_marking and question.teacher_marking.visible_symbols) for question in questions),
        questions_with_individual_mark=sum(bool(question.teacher_marking and question.teacher_marking.visible_individual_score is not None) for question in questions),
        identity_fields_present=[field for field, value in identity.items() if value is not None],
    )
    checks: list[EvidenceCheck] = []
    blockers: list[str] = []
    warnings: list[str] = []
    if not sections:
        blockers.append("no_visible_sections")
        checks.append(EvidenceCheck(name="section_presence", status="fail", reason="no_visible_sections", explanation="No visible sections were extracted."))
    else:
        checks.append(EvidenceCheck(name="section_presence", status="pass", explanation=f"{len(sections)} visible section(s) extracted."))
    if reported is None:
        warnings.append("reported_total_missing")
        checks.append(EvidenceCheck(name="worksheet_reported_score", status="warning", reason="reported_total_missing", explanation="No visible worksheet-reported score was extracted."))
    else:
        checks.append(EvidenceCheck(name="worksheet_reported_score", status="pass", explanation=f"Visible worksheet-reported score: {reported.obtained}" + (f"/{reported.maximum}" if reported.maximum is not None else ".")))
    if sections and len(section_scores) != len(sections):
        warnings.append("section_scores_incomplete")
        checks.append(EvidenceCheck(name="section_score_coverage", status="warning", reason="section_scores_incomplete", explanation=f"Visible section scores are present for {len(section_scores)} of {len(sections)} section(s)."))
    elif sections:
        checks.append(EvidenceCheck(name="section_score_coverage", status="pass", explanation=f"Visible section scores are present for all {len(sections)} section(s)."))
    if summary.visible_section_score_sum is not None and reported is not None:
        if abs(summary.visible_section_score_sum - reported.obtained) <= 1e-9:
            checks.append(EvidenceCheck(name="section_total_comparison", status="pass", explanation=f"Visible section scores sum to {_display_number(summary.visible_section_score_sum)}, matching the worksheet-reported score."))
        else:
            warnings.append("section_total_mismatch")
            checks.append(EvidenceCheck(name="section_total_comparison", status="fail", reason="section_total_mismatch", explanation=f"Visible section scores sum to {_display_number(summary.visible_section_score_sum)}, while the worksheet reports {_display_number(reported.obtained)}."))
    if "section_total_mismatch" in warnings:
        status: Literal["consistent", "needs_review", "insufficient_evidence"] = "needs_review"
    elif blockers or "reported_total_missing" in warnings or "section_scores_incomplete" in warnings:
        status = "insufficient_evidence"
    else:
        status = "consistent"
    return VerificationResult(status=status, checks=checks, blockers=blockers, warnings=warnings, evidence_summary=summary)
