import json

import pytest

from core.local_result_cache import LocalGroqResultCache
from core.models import VisionExtraction
from core.partial_credit import semantic_partial_credit_suggestions
from core.reference_models import QuestionPaperExtraction, RubricExtraction
from core.rubric_import import (
    IMPORTED_VERIFIED_LIVE_GROQ_PROVIDER,
    VerifiedRubricImportError,
    import_verified_live_document_json,
    import_verified_live_rubric_json,
)


def _payload(identifier="1"):
    return {
        "worksheet_title": "Rubric",
        "sections": [{"questions": [{
            "identifier": identifier,
            "visible_criteria": ["Visible criterion"],
            "visible_maximum_marks": 2,
        }]}],
    }


def test_imported_verified_rubric_is_validated_cached_and_has_non_secret_provenance(tmp_path):
    source = tmp_path / "verified_rubric.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    cache = LocalGroqResultCache(tmp_path / "cache")

    result = import_verified_live_rubric_json(
        source, model="qwen/test", policy_version="policy-v1", cache=cache
    )

    assert result.document_type == "rubric"
    assert result.provenance.provider == IMPORTED_VERIFIED_LIVE_GROQ_PROVIDER
    assert result.provenance.source_image_identifier == "verified_rubric.json"
    assert result.provenance.model == "qwen/test"
    assert result.provenance.policy_version == "policy-v1"
    assert result.result_payload["sections"][0]["questions"][0]["identifier"] == "1"
    assert cache.latest_for_document_type("rubric", "qwen/test", "policy-v1") is not None


@pytest.mark.parametrize("content", ["not json", '{"sections": [{"questions": [{"visible_maximum_marks": -1}]}]}'])
def test_import_rejects_invalid_json_or_model_mismatch(tmp_path, content):
    source = tmp_path / "invalid.json"
    source.write_text(content, encoding="utf-8")
    with pytest.raises(VerifiedRubricImportError):
        import_verified_live_rubric_json(source, model="qwen/test", policy_version="policy-v1", cache=LocalGroqResultCache(tmp_path / "cache"))


def test_imported_rubric_participates_in_three_document_semantic_pipeline(tmp_path):
    source = tmp_path / "verified_rubric.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    cache = LocalGroqResultCache(tmp_path / "cache")
    imported = import_verified_live_rubric_json(source, model="qwen/test", policy_version="policy-v1", cache=cache)
    rubric = RubricExtraction.model_validate(imported.result_payload)
    question_paper = QuestionPaperExtraction.model_validate({"sections": [{"questions": [{"identifier": "1", "printed_text": "Printed question"}]}]})
    answer_sheet = VisionExtraction.model_validate({"sections": [{"questions": [{"identifier": "Q1", "student_answer": {"visible_text": "Visible answer"}, "teacher_marking": {"visible_individual_score": {"obtained": 1}}}]}]})

    class Encoder:
        def encode(self, sentences, *, normalize_embeddings):
            return [[1.0, 0.0] for _ in sentences]

    from core.semantic_matching import SemanticSentenceMatcher
    suggestion = semantic_partial_credit_suggestions(
        question_paper, rubric, answer_sheet,
        matcher=SemanticSentenceMatcher(encoder=Encoder()),
    )[0]
    assert suggestion.question_identifier == "Q1"
    assert suggestion.teacher_awarded_mark == 1
    assert suggestion.suggested_mark == 2


def test_import_accepts_typed_question_paper_and_answer_sheet_provider_dtos(tmp_path):
    question_source = tmp_path / "question.json"
    question_source.write_text(json.dumps({"worksheet_title": "Paper", "sections": [{"questions": [{"identifier": "1", "printed_text": "Question"}]}]}), encoding="utf-8")
    answer_source = tmp_path / "answer.json"
    answer_source.write_text(json.dumps({"student_name": "Student", "sections": [{"questions": [{"identifier": "1", "student_answer": "Answer", "awarded_mark": 2}]}]}), encoding="utf-8")
    cache = LocalGroqResultCache(tmp_path / "cache")

    paper = import_verified_live_document_json(question_source, "question_paper", model="qwen/test", policy_version="policy-v1", cache=cache)
    answer = import_verified_live_document_json(answer_source, "answer_sheet", model="qwen/test", policy_version="policy-v1", cache=cache)

    assert paper.document_type == "question_paper"
    assert paper.result_payload["sections"][0]["questions"][0]["printed_text"] == "Question"
    assert answer.document_type == "student_answer_sheet"
    assert answer.result_payload["student"]["name"] == "Student"
