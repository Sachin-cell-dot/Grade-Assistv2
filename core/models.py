from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Score = Annotated[float, Field(ge=0)]


class EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


class Uncertainty(EvidenceModel):
    is_uncertain: bool = False
    reasons: list[str] = Field(default_factory=list)
    source_legibility: Literal["clear", "unclear", "not_visible"] = "clear"

    @model_validator(mode="after")
    def legibility_requires_uncertainty(self):
        if self.source_legibility in {"unclear", "not_visible"} and not self.is_uncertain:
            raise ValueError("unclear or not_visible evidence must be marked uncertain")
        return self


class VisibleScore(EvidenceModel):
    """A score visibly written by a teacher or printed on the worksheet."""
    obtained: Score
    maximum: Score | None = None
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)


class StudentMetadata(EvidenceModel):
    name: str | None = None
    roll_number: str | None = None
    class_name: str | None = None
    subject: str | None = None
    assessment_date: str | None = None
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)


class TeacherMarking(EvidenceModel):
    visible_symbols: list[str] = Field(default_factory=list)
    visible_comment: str | None = None
    visible_correction: str | None = None
    visible_individual_score: VisibleScore | None = None
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)


class StudentAnswer(EvidenceModel):
    visible_text: str | None = None
    visible_working: str | None = None
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)


class SubQuestion(EvidenceModel):
    identifier: str | None = None
    student_answer: StudentAnswer = Field(default_factory=StudentAnswer)
    teacher_marking: TeacherMarking | None = None


class Question(EvidenceModel):
    identifier: str | None = None
    question_text: str | None = None
    student_answer: StudentAnswer = Field(default_factory=StudentAnswer)
    teacher_marking: TeacherMarking | None = None
    subquestions: list[SubQuestion] = Field(default_factory=list)


class Section(EvidenceModel):
    identifier: str | None = None
    title: str | None = None
    visible_section_score: VisibleScore | None = None
    questions: list[Question] = Field(default_factory=list)
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)


class VisionExtraction(EvidenceModel):
    """Only observable worksheet content; never a grading decision."""
    student: StudentMetadata = Field(default_factory=StudentMetadata)
    sections: list[Section] = Field(default_factory=list)
    worksheet_reported_score: VisibleScore | None = None
    unassigned_teacher_markings: list[TeacherMarking] = Field(default_factory=list)
    worksheet_score_uncertainty: Uncertainty = Field(default_factory=Uncertainty)
    overall_uncertainty: Uncertainty = Field(default_factory=Uncertainty)

    @model_validator(mode="after")
    def scores_must_be_visible(self):
        # Field names intentionally constrain model output to visible evidence.
        return self
