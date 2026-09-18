"""Identifier-only mapping across independently extracted evidence documents."""
from pydantic import Field

from core.models import EvidenceModel, Question, VisionExtraction
from core.reference_models import PrintedQuestion, QuestionPaperExtraction, RubricExtraction, RubricQuestion


class QuestionEvidenceMapping(EvidenceModel):
    question_identifier: str | None = None
    answer_question: Question
    question_paper_question: PrintedQuestion | None = None
    rubric_question: RubricQuestion | None = None
    mapping_issues: list[str] = Field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return not self.mapping_issues and self.question_paper_question is not None and self.rubric_question is not None


def normalise_question_identifier(identifier: str | None) -> str:
    value = "".join(character for character in (identifier or "").lower() if character.isalnum())
    return value[1:] if value.startswith("q") and value[1:].isdigit() else value


def map_question_evidence(question_paper: QuestionPaperExtraction, rubric: RubricExtraction, answer_sheet: VisionExtraction) -> list[QuestionEvidenceMapping]:
    """Map only exact normalized visible identifiers; never infer missing associations."""
    paper_by_id = {
        normalise_question_identifier(question.identifier): question
        for section in question_paper.sections
        for question in section.questions
        if normalise_question_identifier(question.identifier)
    }
    rubric_by_id = {
        normalise_question_identifier(question.identifier): question
        for section in rubric.sections
        for question in section.questions
        if normalise_question_identifier(question.identifier)
    }
    mappings = []
    for section in answer_sheet.sections:
        for answer in section.questions:
            normalized = normalise_question_identifier(answer.identifier)
            issues = []
            if not normalized:
                issues.append("answer_question_identifier_missing")
            if normalized and normalized not in paper_by_id:
                issues.append("question_paper_identifier_unmapped")
            if normalized and normalized not in rubric_by_id:
                issues.append("rubric_identifier_unmapped")
            mappings.append(QuestionEvidenceMapping(
                question_identifier=answer.identifier,
                answer_question=answer,
                question_paper_question=paper_by_id.get(normalized),
                rubric_question=rubric_by_id.get(normalized),
                mapping_issues=issues,
            ))
    return mappings
