"""Local, explainable partial-credit suggestions. Never changes extracted evidence."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum

from pydantic import Field

from core.evidence_mapping import map_question_evidence
from core.models import EvidenceModel, Question, VisionExtraction
from core.reference_models import QuestionPaperExtraction, RubricExtraction, RubricQuestion
from core.semantic_matching import (
    SEMANTIC_METHOD_VERSION,
    SemanticEmbeddingConfig,
    SemanticModelUnavailable,
    SemanticSentenceMatcher,
)

METHOD_VERSION = "deterministic-lexical-concept-v1"
_WORD = re.compile(r"[a-zA-Z]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "by", "for", "from", "give", "he", "her", "his", "in", "is",
    "it", "of", "on", "or", "one", "that", "the", "their", "this", "to", "what", "with", "was", "were",
    "identifies", "identify", "mentions", "mention", "says", "states", "provides", "explains", "describes",
}
# Generic English word families only; no worksheet-specific answers, names, or question IDs.
_SYNONYM_FAMILIES = (
    {"help", "helps", "helped", "helpful", "assist", "assisted", "support", "supported", "aid"},
    {"happy", "happiness", "pleased", "glad", "satisfied"},
    {"worry", "worried", "concern", "concerned", "anxious", "uncertain"},
    {"kind", "kindness", "compassion", "compassionate", "caring", "helpfulness"},
    {"travel", "travelling", "traveled", "journey", "arrive", "arrival"},
)


class SuggestionStatus(str, Enum):
    AGREES_WITH_TEACHER = "AGREES_WITH_TEACHER"
    SUGGEST_REVIEW = "SUGGEST_REVIEW"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class TeacherDispositionAction(str, Enum):
    ACCEPT_SUGGESTION = "ACCEPT_SUGGESTION"
    EDIT_FINAL_MARK = "EDIT_FINAL_MARK"
    PROCEED_WITH_TEACHER_MARK = "PROCEED_WITH_TEACHER_MARK"


class RubricCriterion(EvidenceModel):
    text: str
    weight: float = Field(gt=0)


class CriterionMatchResult(EvidenceModel):
    criterion: RubricCriterion
    matched: bool
    similarity_score: float = Field(ge=0, le=1)
    matched_answer_phrase: str | None = None
    matched_concepts: list[str] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)
    method: str = METHOD_VERSION
    model_name: str | None = None
    threshold_used: float | None = Field(default=None, ge=0, le=1)


class PartialCreditSuggestion(EvidenceModel):
    question_identifier: str | None = None
    teacher_awarded_mark: float | None = Field(default=None, ge=0)
    rubric_maximum_marks: float | None = Field(default=None, ge=0)
    suggested_mark: float | None = Field(default=None, ge=0)
    matched_rubric_criteria: list[CriterionMatchResult] = Field(default_factory=list)
    missing_rubric_criteria: list[CriterionMatchResult] = Field(default_factory=list)
    evidence_phrases: list[str] = Field(default_factory=list)
    matching_score: float | None = Field(default=None, ge=0, le=1)
    matching_method: str = METHOD_VERSION
    matching_mode: str = "lexical_fallback"
    rationale: str
    status: SuggestionStatus


class TeacherDisposition(EvidenceModel):
    question_identifier: str | None = None
    action: TeacherDispositionAction
    original_teacher_mark: float | None = Field(default=None, ge=0)
    system_suggested_mark: float = Field(ge=0)
    teacher_final_mark: float = Field(ge=0)
    rationale: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LexicalConceptConfig(EvidenceModel):
    semantic_threshold: float = Field(default=0.5, ge=0, le=1)
    short_answer_maximum_marks: float = Field(default=2, gt=0)


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def _concept(token: str) -> str:
    stemmed = _stem(token.lower())
    for family in _SYNONYM_FAMILIES:
        if stemmed in {_stem(word) for word in family}:
            return min(family)
    return stemmed


def _concepts(text: str) -> list[str]:
    return sorted({_concept(word) for word in _WORD.findall(text.lower()) if word not in _STOPWORDS})


def _sentences(answer: str) -> list[str]:
    return [sentence.strip() for sentence in _SENTENCE.split(answer) if sentence.strip()] or [answer.strip()]


def _best_match(criterion: RubricCriterion, answer: str) -> CriterionMatchResult:
    criterion_concepts = _concepts(criterion.text)
    if not criterion_concepts:
        return CriterionMatchResult(criterion=criterion, matched=False, similarity_score=0, missing_concepts=[])
    best_phrase = None
    best_matched: set[str] = set()
    for sentence in _sentences(answer):
        matched = set(criterion_concepts) & set(_concepts(sentence))
        if len(matched) > len(best_matched):
            best_phrase, best_matched = sentence, matched
    score = len(best_matched) / len(criterion_concepts)
    return CriterionMatchResult(
        criterion=criterion,
        matched=False,
        similarity_score=score,
        matched_answer_phrase=best_phrase if best_matched else None,
        matched_concepts=sorted(best_matched),
        missing_concepts=sorted(set(criterion_concepts) - best_matched),
    )


def _teacher_mark(question: Question) -> float | None:
    marking = question.teacher_marking
    return marking.visible_individual_score.obtained if marking and marking.visible_individual_score else None


def _is_long_answer(answer: str, maximum: float, config: LexicalConceptConfig) -> bool:
    return maximum > config.short_answer_maximum_marks or len(_sentences(answer)) > 1


def suggest_partial_credit(question: Question, rubric_question: RubricQuestion | None, config: LexicalConceptConfig | None = None) -> PartialCreditSuggestion:
    """Suggest a mark from local lexical coverage; never changes teacher evidence."""
    config = config or LexicalConceptConfig()
    answer = question.student_answer.visible_text
    teacher_mark = _teacher_mark(question)
    if rubric_question is None or not rubric_question.visible_criteria or rubric_question.visible_maximum_marks is None:
        return PartialCreditSuggestion(question_identifier=question.identifier, teacher_awarded_mark=teacher_mark, rationale="No usable visible rubric criteria and maximum mark are available. Teacher review is required.", status=SuggestionStatus.INSUFFICIENT_EVIDENCE)
    if not answer or not answer.strip():
        return PartialCreditSuggestion(question_identifier=question.identifier, teacher_awarded_mark=teacher_mark, rubric_maximum_marks=rubric_question.visible_maximum_marks, rationale="No extracted student answer is available for comparison. Teacher review is required.", status=SuggestionStatus.INSUFFICIENT_EVIDENCE)
    maximum = rubric_question.visible_maximum_marks
    criterion_texts = [text for text in rubric_question.visible_criteria if text.strip()]
    criteria = [RubricCriterion(text=text, weight=maximum / len(criterion_texts)) for text in criterion_texts]
    if not criteria:
        return PartialCreditSuggestion(question_identifier=question.identifier, teacher_awarded_mark=teacher_mark, rubric_maximum_marks=maximum, rationale="No usable visible rubric criteria are available. Teacher review is required.", status=SuggestionStatus.INSUFFICIENT_EVIDENCE)
    long_answer = _is_long_answer(answer, maximum, config)
    matches = []
    for criterion in criteria:
        result = _best_match(criterion, answer)
        matches.append(result.model_copy(update={"matched": result.similarity_score >= config.semantic_threshold}))
    proposed = sum(match.criterion.weight * (match.similarity_score if long_answer else float(match.matched)) for match in matches)
    proposed = min(maximum, max(0, round(proposed, 2)))
    matched = [match for match in matches if match.matched]
    missing = [match for match in matches if not match.matched]
    status = SuggestionStatus.INSUFFICIENT_EVIDENCE if teacher_mark is None else (SuggestionStatus.AGREES_WITH_TEACHER if abs(teacher_mark - proposed) <= 1e-9 else SuggestionStatus.SUGGEST_REVIEW)
    evidence_phrases = list(dict.fromkeys(match.matched_answer_phrase for match in matched if match.matched_answer_phrase))
    rationale = f"GradeAssist found rubric evidence supporting {_display(proposed)} of {_display(maximum)} available marks. Teacher review is required before any final decision."
    return PartialCreditSuggestion(
        question_identifier=question.identifier,
        teacher_awarded_mark=teacher_mark,
        rubric_maximum_marks=maximum,
        suggested_mark=proposed,
        matched_rubric_criteria=matched,
        missing_rubric_criteria=missing,
        evidence_phrases=evidence_phrases,
        matching_score=round(sum(match.similarity_score for match in matches) / len(matches), 3),
        rationale=rationale,
        status=status,
    )


def _display(value: float) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


def _normalise_identifier(identifier: str | None) -> str:
    text = "".join(character for character in (identifier or "").lower() if character.isalnum())
    return text[1:] if text.startswith("q") and text[1:].isdigit() else text


def partial_credit_suggestions(answer_sheet: VisionExtraction, rubric: RubricExtraction, config: LexicalConceptConfig | None = None) -> list[PartialCreditSuggestion]:
    rubric_by_identifier = {
        _normalise_identifier(question.identifier): question
        for section in rubric.sections
        for question in section.questions
        if _normalise_identifier(question.identifier)
    }
    return [
        suggest_partial_credit(question, rubric_by_identifier.get(_normalise_identifier(question.identifier)), config)
        for section in answer_sheet.sections
        for question in section.questions
    ]


def _semantic_suggestion(question: Question, rubric_question: RubricQuestion, matcher: SemanticSentenceMatcher) -> PartialCreditSuggestion:
    answer = question.student_answer.visible_text
    teacher_mark = _teacher_mark(question)
    maximum = rubric_question.visible_maximum_marks
    if not answer or not answer.strip() or maximum is None or not rubric_question.visible_criteria:
        return PartialCreditSuggestion(question_identifier=question.identifier, teacher_awarded_mark=teacher_mark, rubric_maximum_marks=maximum, matching_method=SEMANTIC_METHOD_VERSION, matching_mode="semantic_embedding", rationale="Required visible answer or rubric evidence is unavailable. Teacher review is required.", status=SuggestionStatus.INSUFFICIENT_EVIDENCE)
    criterion_texts = [text for text in rubric_question.visible_criteria if text.strip()]
    if not criterion_texts:
        return PartialCreditSuggestion(question_identifier=question.identifier, teacher_awarded_mark=teacher_mark, rubric_maximum_marks=maximum, matching_method=SEMANTIC_METHOD_VERSION, matching_mode="semantic_embedding", rationale="No usable visible rubric criteria are available. Teacher review is required.", status=SuggestionStatus.INSUFFICIENT_EVIDENCE)
    criteria = [RubricCriterion(text=text, weight=maximum / len(criterion_texts)) for text in criterion_texts]
    long_answer = _is_long_answer(answer, maximum, LexicalConceptConfig())
    matches = []
    for criterion in criteria:
        semantic = matcher.match(criterion.text, answer)
        matches.append(CriterionMatchResult(
            criterion=criterion,
            matched=semantic.matched,
            similarity_score=semantic.similarity_score,
            matched_answer_phrase=semantic.matched_answer_phrase,
            method=semantic.method,
            model_name=semantic.model_name,
            threshold_used=semantic.threshold_used,
        ))
    proposed = sum(match.criterion.weight * (match.similarity_score if long_answer else float(match.matched)) for match in matches)
    proposed = min(maximum, max(0, round(proposed, 2)))
    matched = [match for match in matches if match.matched]
    missing = [match for match in matches if not match.matched]
    status = SuggestionStatus.INSUFFICIENT_EVIDENCE if teacher_mark is None else (SuggestionStatus.AGREES_WITH_TEACHER if abs(teacher_mark - proposed) <= 1e-9 else SuggestionStatus.SUGGEST_REVIEW)
    phrases = list(dict.fromkeys(match.matched_answer_phrase for match in matched if match.matched_answer_phrase))
    rationale = f"GradeAssist found rubric evidence supporting {_display(proposed)} of {_display(maximum)} available marks. Teacher review is required before any final decision."
    return PartialCreditSuggestion(
        question_identifier=question.identifier,
        teacher_awarded_mark=teacher_mark,
        rubric_maximum_marks=maximum,
        suggested_mark=proposed,
        matched_rubric_criteria=matched,
        missing_rubric_criteria=missing,
        evidence_phrases=phrases,
        matching_score=round(sum(match.similarity_score for match in matches) / len(matches), 3),
        matching_method=SEMANTIC_METHOD_VERSION,
        matching_mode="semantic_embedding",
        rationale=rationale,
        status=status,
    )


def semantic_partial_credit_suggestions(
    question_paper: QuestionPaperExtraction,
    rubric: RubricExtraction,
    answer_sheet: VisionExtraction,
    config: SemanticEmbeddingConfig | None = None,
    matcher: SemanticSentenceMatcher | None = None,
) -> list[PartialCreditSuggestion]:
    """Primary three-source semantic workflow, mapped only by visible question identifiers."""
    config = config or SemanticEmbeddingConfig()
    mappings = map_question_evidence(question_paper, rubric, answer_sheet)
    try:
        matcher = matcher or SemanticSentenceMatcher(config)
    except SemanticModelUnavailable:
        if not config.allow_lexical_fallback:
            raise
        fallback_config = LexicalConceptConfig(semantic_threshold=config.semantic_threshold)
        results = []
        for mapping in mappings:
            if not mapping.is_complete:
                results.append(PartialCreditSuggestion(question_identifier=mapping.question_identifier, teacher_awarded_mark=_teacher_mark(mapping.answer_question), rationale=f"Question identifier mapping is incomplete: {', '.join(mapping.mapping_issues)}. Teacher review is required.", status=SuggestionStatus.INSUFFICIENT_EVIDENCE, matching_mode="lexical_fallback"))
            else:
                fallback = suggest_partial_credit(mapping.answer_question, mapping.rubric_question, fallback_config)
                results.append(fallback.model_copy(update={"matching_mode": "lexical_fallback", "rationale": f"Offline lexical fallback used because the local semantic model was unavailable. {fallback.rationale}"}))
        return results
    results = []
    for mapping in mappings:
        if not mapping.is_complete:
            results.append(PartialCreditSuggestion(question_identifier=mapping.question_identifier, teacher_awarded_mark=_teacher_mark(mapping.answer_question), rationale=f"Question identifier mapping is incomplete: {', '.join(mapping.mapping_issues)}. Teacher review is required.", status=SuggestionStatus.INSUFFICIENT_EVIDENCE, matching_method=SEMANTIC_METHOD_VERSION, matching_mode="semantic_embedding"))
            continue
        results.append(_semantic_suggestion(mapping.answer_question, mapping.rubric_question, matcher))
    # A question present in both reference documents but wholly absent from the
    # answer-sheet extraction is still a factual outcome: there is no answer
    # evidence to compare.  Surface it explicitly rather than silently
    # omitting it or inventing a blank answer/teacher mark.
    answered_identifiers = {
        _normalise_identifier(mapping.question_identifier)
        for mapping in mappings
        if _normalise_identifier(mapping.question_identifier)
    }
    paper_by_identifier = {
        _normalise_identifier(question.identifier): question
        for section in question_paper.sections
        for question in section.questions
        if _normalise_identifier(question.identifier)
    }
    rubric_by_identifier = {
        _normalise_identifier(question.identifier): question
        for section in rubric.sections
        for question in section.questions
        if _normalise_identifier(question.identifier)
    }
    for identifier in paper_by_identifier.keys() & rubric_by_identifier.keys() - answered_identifiers:
        rubric_question = rubric_by_identifier[identifier]
        results.append(PartialCreditSuggestion(
            question_identifier=paper_by_identifier[identifier].identifier,
            rubric_maximum_marks=rubric_question.visible_maximum_marks,
            matching_method=SEMANTIC_METHOD_VERSION,
            matching_mode="semantic_embedding",
            rationale="No extracted student answer is available for this mapped question. Teacher review is required.",
            status=SuggestionStatus.INSUFFICIENT_EVIDENCE,
        ))
    return results


def record_teacher_disposition(suggestion: PartialCreditSuggestion, action: TeacherDispositionAction, rationale: str | None = None, edited_final_mark: float | None = None) -> TeacherDisposition:
    if suggestion.suggested_mark is None or suggestion.rubric_maximum_marks is None:
        raise ValueError("A final mark cannot be recorded without a scored suggestion.")
    if action == TeacherDispositionAction.ACCEPT_SUGGESTION:
        final_mark = suggestion.suggested_mark
    elif action == TeacherDispositionAction.PROCEED_WITH_TEACHER_MARK:
        if suggestion.teacher_awarded_mark is None:
            raise ValueError("No visible teacher mark is available to proceed with.")
        final_mark = suggestion.teacher_awarded_mark
    else:
        if edited_final_mark is None or edited_final_mark < 0 or edited_final_mark > suggestion.rubric_maximum_marks:
            raise ValueError("Edited final mark must be within the visible rubric maximum.")
        final_mark = edited_final_mark
    return TeacherDisposition(question_identifier=suggestion.question_identifier, action=action, original_teacher_mark=suggestion.teacher_awarded_mark, system_suggested_mark=suggestion.suggested_mark, teacher_final_mark=final_mark, rationale=rationale)


def append_teacher_disposition(existing: list[TeacherDisposition], disposition: TeacherDisposition) -> list[TeacherDisposition]:
    """Return a new append-only disposition history; prior records are never mutated."""
    return [*existing, disposition]
