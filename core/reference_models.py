"""Typed visible-reference documents. These are not grading decisions."""
from pydantic import Field

from core.models import EvidenceModel, Uncertainty


class PrintedQuestion(EvidenceModel):
    identifier: str | None = None
    printed_text: str | None = None
    printed_maximum_marks: float | None = Field(default=None, ge=0)
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)


class QuestionPaperSection(EvidenceModel):
    identifier: str | None = None
    title: str | None = None
    printed_maximum_marks: float | None = Field(default=None, ge=0)
    questions: list[PrintedQuestion] = Field(default_factory=list)
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)


class QuestionPaperExtraction(EvidenceModel):
    worksheet_title: str | None = None
    subject: str | None = None
    class_name: str | None = None
    total_marks: float | None = Field(default=None, ge=0)
    printed_passage: str | None = None
    sections: list[QuestionPaperSection] = Field(default_factory=list)
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)


class RubricQuestion(EvidenceModel):
    identifier: str | None = None
    visible_criteria: list[str] = Field(default_factory=list)
    # Optional only: a teacher/imported rubric may preserve explicitly visible
    # per-criterion marks. Groq's existing rubric DTO remains unchanged.
    visible_criterion_weights: list[float] = Field(default_factory=list)
    visible_maximum_marks: float | None = Field(default=None, ge=0)
    visible_key_context: str | None = None
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)


class RubricSection(EvidenceModel):
    identifier: str | None = None
    title: str | None = None
    questions: list[RubricQuestion] = Field(default_factory=list)
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)


class RubricExtraction(EvidenceModel):
    worksheet_title: str | None = None
    subject: str | None = None
    sections: list[RubricSection] = Field(default_factory=list)
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)
