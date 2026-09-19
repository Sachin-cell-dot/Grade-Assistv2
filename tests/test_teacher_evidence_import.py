import json

import pytest

from core.local_result_cache import LocalGroqResultCache
from core.partial_credit import semantic_partial_credit_suggestions
from core.reference_models import QuestionPaperExtraction, RubricExtraction
from core.semantic_matching import SemanticSentenceMatcher
from core.teacher_evidence_import import (
    TEACHER_VERIFIED_LABEL,
    TEACHER_VERIFIED_MODEL,
    TEACHER_VERIFIED_SOURCE_TYPE,
    TeacherEvidenceImportError,
    import_teacher_verified_answer_evidence,
    import_prior_successful_groq_transcript,
    local_result_label,
)
from core.models import VisionExtraction


def _evidence():
    return {"sections": [{"questions": [{"identifier": "Q1", "student_answer": {"visible_text": "Visible answer"}, "teacher_marking": {"visible_individual_score": {"obtained": 1}}}]}]}


def test_teacher_verified_import_validates_isolated_cache_and_labels_provenance(tmp_path):
    source = tmp_path / "answer.json"
    source.write_text(json.dumps(_evidence()), encoding="utf-8")
    cache = LocalGroqResultCache(tmp_path / "cache")
    result = import_teacher_verified_answer_evidence(source, source_image_filename="worksheet.jpeg", note="Teacher checked evidence", policy_version="policy-v1", cache=cache)
    assert result.document_type == "student_answer_sheet"
    assert result.model == TEACHER_VERIFIED_MODEL
    assert result.provenance.source_type == TEACHER_VERIFIED_SOURCE_TYPE
    assert result.provenance.import_note == "Teacher checked evidence"
    assert local_result_label(result.provenance) == TEACHER_VERIFIED_LABEL
    assert cache.get(source, "student_answer_sheet", TEACHER_VERIFIED_MODEL, "policy-v1") is not None
    assert cache.latest_for_document_type("student_answer_sheet", "unrelated-model", "policy-v1") is None


@pytest.mark.parametrize("payload", ["not json", '{"sections": [{"unknown": 1}]}'])
def test_teacher_verified_import_rejects_invalid_canonical_schema(tmp_path, payload):
    source = tmp_path / "answer.json"
    source.write_text(payload, encoding="utf-8")
    with pytest.raises(TeacherEvidenceImportError):
        import_teacher_verified_answer_evidence(source, source_image_filename="worksheet.jpeg", policy_version="policy-v1", cache=LocalGroqResultCache(tmp_path / "cache"))


def test_imported_evidence_runs_local_semantic_pipeline_without_provider(tmp_path):
    source = tmp_path / "answer.json"
    source.write_text(json.dumps(_evidence()), encoding="utf-8")
    imported = import_teacher_verified_answer_evidence(source, source_image_filename="worksheet.jpeg", policy_version="policy-v1", cache=LocalGroqResultCache(tmp_path / "cache"))
    answer = VisionExtraction.model_validate(imported.result_payload)
    paper = QuestionPaperExtraction.model_validate({"sections": [{"questions": [{"identifier": "1", "printed_text": "Question"}]}]})
    rubric = RubricExtraction.model_validate({"sections": [{"questions": [{"identifier": "1", "visible_criteria": ["Criterion"], "visible_maximum_marks": 2}]}]})

    class Encoder:
        def encode(self, sentences, *, normalize_embeddings): return [[1.0, 0.0] for _ in sentences]

    before = answer.model_dump(mode="json")
    result = semantic_partial_credit_suggestions(paper, rubric, answer, matcher=SemanticSentenceMatcher(encoder=Encoder()))[0]
    assert result.teacher_awarded_mark == 1
    assert answer.model_dump(mode="json") == before


def test_recovered_prior_groq_evidence_is_distinct_and_requires_teacher_review(tmp_path):
    source = tmp_path / "prior.json"
    source.write_text(json.dumps(_evidence()), encoding="utf-8")
    result = import_prior_successful_groq_transcript(source, source_image_filename="worksheet.jpeg", model="qwen/test", policy_version="policy-v1", cache=LocalGroqResultCache(tmp_path / "cache"))
    assert result.provenance.source_type == "prior_successful_groq_transcript"
    assert result.provenance.requires_teacher_review is True
    assert local_result_label(result.provenance) == "Recovered prior Groq evidence"
