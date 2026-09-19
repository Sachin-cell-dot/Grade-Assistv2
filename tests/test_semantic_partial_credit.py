import core.partial_credit as partial_credit
import pytest
from core.models import VisionExtraction
from core.semantic_matching import SemanticEmbeddingConfig, SemanticModelUnavailable, SemanticSentenceMatcher, visible_rubric_aliases
from core.reference_models import QuestionPaperExtraction, RubricExtraction


class FakeEncoder:
    def encode(self, sentences, *, normalize_embeddings):
        vectors = {
            "criterion match": [1.0, 0.0],
            "criterion miss": [0.0, 1.0],
            "answer evidence": [1.0, 0.0],
            "weak evidence": [0.6, 0.8],
        }
        return [vectors[text] for text in sentences]


def sources(answer="answer evidence", teacher_mark=0, criteria=None, maximum=2, answer_id="Q1", paper_id="1", rubric_id="1"):
    criteria = criteria or ["criterion match"]
    question_paper = QuestionPaperExtraction.model_validate({"sections": [{"questions": [{"identifier": paper_id, "printed_text": "Visible printed question"}]}]})
    rubric = RubricExtraction.model_validate({"sections": [{"questions": [{"identifier": rubric_id, "visible_criteria": criteria, "visible_maximum_marks": maximum}]}]})
    answer_sheet = VisionExtraction.model_validate({"sections": [{"questions": [{"identifier": answer_id, "student_answer": {"visible_text": answer}, "teacher_marking": {"visible_individual_score": {"obtained": teacher_mark}}}]}]})
    return question_paper, rubric, answer_sheet


def matcher(threshold=0.55):
    return SemanticSentenceMatcher(SemanticEmbeddingConfig(semantic_threshold=threshold), encoder=FakeEncoder())


def test_identifier_mapping_and_semantic_match_keep_teacher_mark_unchanged():
    question_paper, rubric, answer_sheet = sources(teacher_mark=0)
    before = answer_sheet.model_dump(mode="json")
    suggestion = partial_credit.semantic_partial_credit_suggestions(question_paper, rubric, answer_sheet, matcher=matcher())[0]
    assert suggestion.suggested_mark == 2
    assert suggestion.status == partial_credit.SuggestionStatus.SUGGEST_REVIEW
    assert suggestion.matched_rubric_criteria[0].matched_answer_phrase == "answer evidence"
    assert suggestion.matched_rubric_criteria[0].similarity_score == 1
    assert suggestion.matching_mode == "semantic_embedding"
    assert answer_sheet.model_dump(mode="json") == before


def test_semantic_non_match_and_threshold_behavior():
    question_paper, rubric, answer_sheet = sources(answer="weak evidence", teacher_mark=0)
    no_match = partial_credit.semantic_partial_credit_suggestions(question_paper, rubric, answer_sheet, matcher=matcher(threshold=0.9))[0]
    assert no_match.suggested_mark == 0
    assert no_match.status == partial_credit.SuggestionStatus.AGREES_WITH_TEACHER
    matched = partial_credit.semantic_partial_credit_suggestions(question_paper, rubric, answer_sheet, matcher=matcher(threshold=0.5))[0]
    assert matched.suggested_mark == 2


def test_weighted_partial_coverage_and_mark_cap():
    question_paper, rubric, answer_sheet = sources(criteria=["criterion match", "criterion miss"], maximum=5)
    suggestion = partial_credit.semantic_partial_credit_suggestions(question_paper, rubric, answer_sheet, matcher=matcher())[0]
    assert suggestion.suggested_mark == 2.5
    assert suggestion.suggested_mark <= suggestion.rubric_maximum_marks
    assert len(suggestion.missing_rubric_criteria) == 1


def test_missing_mapping_or_evidence_is_insufficient():
    question_paper, rubric, answer_sheet = sources(paper_id="2")
    suggestion = partial_credit.semantic_partial_credit_suggestions(question_paper, rubric, answer_sheet, matcher=matcher())[0]
    assert suggestion.status == partial_credit.SuggestionStatus.INSUFFICIENT_EVIDENCE
    assert "question_paper_identifier_unmapped" in suggestion.rationale


def test_reference_mapped_question_without_student_answer_is_explicitly_insufficient():
    question_paper = QuestionPaperExtraction.model_validate({"sections": [{"questions": [{"identifier": "7", "printed_text": "Question seven"}]}]})
    rubric = RubricExtraction.model_validate({"sections": [{"questions": [{"identifier": "7", "visible_criteria": ["Visible criterion"], "visible_maximum_marks": 5}]}]})
    answer_sheet = VisionExtraction.model_validate({"sections": []})

    suggestion = partial_credit.semantic_partial_credit_suggestions(question_paper, rubric, answer_sheet, matcher=matcher())[0]

    assert suggestion.question_identifier == "7"
    assert suggestion.status == partial_credit.SuggestionStatus.INSUFFICIENT_EVIDENCE
    assert suggestion.teacher_awarded_mark is None
    assert suggestion.suggested_mark is None


def test_lexical_fallback_is_never_presented_as_semantic(monkeypatch):
    question_paper, rubric, answer_sheet = sources()

    def unavailable(config):
        raise SemanticModelUnavailable("offline")

    monkeypatch.setattr(partial_credit, "SemanticSentenceMatcher", unavailable)
    suggestion = partial_credit.semantic_partial_credit_suggestions(question_paper, rubric, answer_sheet, SemanticEmbeddingConfig(allow_lexical_fallback=True))[0]
    assert suggestion.matching_mode == "lexical_fallback"
    assert "Offline lexical fallback" in suggestion.rationale


def test_semantic_module_has_no_groq_or_ollama_dependency():
    source = open("core/semantic_matching.py", encoding="utf-8").read()
    assert "Groq" not in source
    assert "Ollama" not in source


@pytest.mark.parametrize(
    ("criterion", "answer"),
    [
        ("Says that Meera stayed with/accompanied the man.", "She accompanied him at the station."),
        ("Says that she helped him carry his heavy bag.", "She carried his luggage."),
        ("Identifies that Meera felt happy/satisfied after helping the man.", "She felt pleased after helping."),
        ("Connects her happiness to the positive effect of her kindness.", "Helping made a positive difference."),
        ("Identifies that heavy rain made him uncertain / worried about the train.", "He was concerned about the train."),
    ],
)
def test_visible_rubric_aliases_cover_clear_equivalences(criterion, answer):
    assert visible_rubric_aliases(criterion, answer)


def test_visible_rubric_aliases_do_not_match_unrelated_answer():
    assert visible_rubric_aliases("Says that she helped him carry his heavy bag.", "The classroom was quiet.") == []
    assert visible_rubric_aliases("Recognizes the positive emotional effect of kindness.", "Kindness can help others.") == []


def test_alias_supported_long_answer_uses_only_that_criterion_full_weight():
    question_paper, rubric, answer_sheet = sources(
        answer="She accompanied him.", teacher_mark=0,
        criteria=["Says that Meera stayed with/accompanied the man.", "criterion miss"], maximum=4,
    )
    class AliasEncoder:
        def encode(self, sentences, *, normalize_embeddings):
            return [[0.0, 1.0] if text == "criterion miss" else [1.0, 0.0] for text in sentences]

    suggestion = partial_credit.semantic_partial_credit_suggestions(question_paper, rubric, answer_sheet, matcher=SemanticSentenceMatcher(encoder=AliasEncoder()))[0]
    assert suggestion.suggested_mark == 2


@pytest.mark.parametrize(("raw", "maximum", "expected"), [(4.02, 5, 4.0), (4.25, 5, 4.5), (9, 5, 5.0), (0, 5, 0.0)])
def test_teacher_facing_semantic_mark_rounds_to_half_and_caps(raw, maximum, expected):
    assert partial_credit.displayed_half_mark(raw, maximum) == expected


def test_raw_semantic_coverage_is_preserved_while_displayed_mark_routes_teacher_review():
    question_paper, rubric, answer_sheet = sources(teacher_mark=4, criteria=["criterion match"], maximum=5)

    class CoverageEncoder:
        def encode(self, sentences, *, normalize_embeddings):
            return [[1.0, 0.0], [0.804, 0.0]]

    suggestion = partial_credit.semantic_partial_credit_suggestions(question_paper, rubric, answer_sheet, matcher=SemanticSentenceMatcher(encoder=CoverageEncoder()))[0]
    assert suggestion.raw_semantic_coverage == 4.02
    assert suggestion.suggested_mark == 4.0
    assert suggestion.status == partial_credit.SuggestionStatus.AGREES_WITH_TEACHER


def test_missing_evidence_has_no_raw_semantic_coverage():
    question_paper, rubric, answer_sheet = sources(answer="")
    suggestion = partial_credit.semantic_partial_credit_suggestions(question_paper, rubric, answer_sheet, matcher=matcher())[0]
    assert suggestion.suggested_mark is None
    assert suggestion.raw_semantic_coverage is None
