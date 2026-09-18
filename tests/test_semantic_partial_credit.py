import core.partial_credit as partial_credit
from core.models import VisionExtraction
from core.semantic_matching import SemanticEmbeddingConfig, SemanticModelUnavailable, SemanticSentenceMatcher
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
