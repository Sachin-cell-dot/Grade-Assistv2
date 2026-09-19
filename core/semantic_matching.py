"""Local CPU sentence-embedding matching; no provider or LLM calls."""
from __future__ import annotations

import re
from math import sqrt
from functools import lru_cache
from typing import Protocol

from pydantic import Field

from core.models import EvidenceModel

SEMANTIC_METHOD_VERSION = "sentence-transformers-bge-small-en-v1.5-v1"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_NORMALIZED = re.compile(r"[^a-z0-9]+")
# Generic language equivalences, activated only when their wording appears in
# the visible rubric criterion. They are not question, student, or mark data.
_VISIBLE_RUBRIC_ALIAS_GROUPS = (
    ("luggage", "bag"),
    ("accompanied", "accompany", "stayed with", "stay with"),
    ("pleased", "happy", "satisfied"),
    ("positive difference", "made a difference", "positive effect"),
    ("worried", "concerned", "anxious"),
    ("travelling", "traveling", "travel", "visit"),
    ("daughter",),
    ("carried", "carry", "helped carry", "helped with"),
)
_CHARACTER_ALIASES = ("kind", "kindness", "compassion", "compassionate", "caring", "helpful", "helpfulness")
_ACTION_WORDS = {"noticed", "notice", "stayed", "stay", "helped", "help", "carried", "carry", "approached", "approach", "offered", "offer", "supported", "support", "left", "leave"}


class SentenceEncoder(Protocol):
    def encode(self, sentences: list[str], *, normalize_embeddings: bool) -> object: ...


class SemanticEmbeddingConfig(EvidenceModel):
    model_name: str = DEFAULT_EMBEDDING_MODEL
    semantic_threshold: float = Field(default=0.65, ge=0, le=1)
    allow_lexical_fallback: bool = True


class SemanticMatchEvidence(EvidenceModel):
    matched: bool
    similarity_score: float = Field(ge=0, le=1)
    matched_answer_phrase: str | None = None
    model_name: str
    threshold_used: float = Field(ge=0, le=1)
    method: str = SEMANTIC_METHOD_VERSION
    match_rationale: str | None = None


class SemanticModelUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=2)
def load_sentence_encoder(model_name: str) -> SentenceEncoder:
    """Load once per process on CPU; model weights remain in the local Hugging Face cache."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name, device="cpu")
    except Exception as exc:  # import, local cache, or model load failure
        raise SemanticModelUnavailable(f"Local sentence embedding model could not load: {model_name}") from exc


def answer_sentences(answer: str) -> list[str]:
    return [sentence.strip() for sentence in _SENTENCE.split(answer) if sentence.strip()]


def cosine_similarity(left: object, right: object) -> float:
    """Return bounded cosine similarity for local embedding vectors."""
    left_values = list(left)
    right_values = list(right)
    numerator = sum(float(a) * float(b) for a, b in zip(left_values, right_values))
    left_norm = sqrt(sum(float(value) ** 2 for value in left_values))
    right_norm = sqrt(sum(float(value) ** 2 for value in right_values))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))


def _normalise(text: str) -> str:
    return f" {_NORMALIZED.sub(' ', text.lower()).strip()} "


def _has_phrase(text: str, phrase: str) -> bool:
    return f" {_normalise(phrase).strip()} " in _normalise(text)


def visible_rubric_aliases(criterion: str, answer_phrase: str) -> list[str]:
    """Return only explicit criterion concepts supported by visible aliases."""
    hits = []
    for group in _VISIBLE_RUBRIC_ALIAS_GROUPS:
        if any(_has_phrase(criterion, term) for term in group) and any(_has_phrase(answer_phrase, term) for term in group):
            hits.append(" / ".join(group))
    if (_has_phrase(criterion, "quality") or _has_phrase(criterion, "character")) and any(_has_phrase(criterion, term) for term in _CHARACTER_ALIASES) and any(_has_phrase(answer_phrase, term) for term in _CHARACTER_ALIASES):
        hits.append("character/quality: " + " / ".join(_CHARACTER_ALIASES))
    # "actions" is visible rubric language; two concrete actions in the
    # answer are support for that one criterion, not a whole-question score.
    if _has_phrase(criterion, "actions") or _has_phrase(criterion, "action"):
        answer_words = set(_normalise(answer_phrase).split())
        action_hits = sorted(answer_words & _ACTION_WORDS)
        if len(action_hits) >= 2:
            hits.append("actions: " + ", ".join(action_hits))
    return hits


class SemanticSentenceMatcher:
    def __init__(self, config: SemanticEmbeddingConfig | None = None, encoder: SentenceEncoder | None = None):
        self.config = config or SemanticEmbeddingConfig()
        self.encoder = encoder or load_sentence_encoder(self.config.model_name)

    def match(self, criterion: str, answer: str) -> SemanticMatchEvidence:
        sentences = answer_sentences(answer)
        if not criterion.strip() or not sentences:
            return SemanticMatchEvidence(
                matched=False,
                similarity_score=0.0,
                model_name=self.config.model_name,
                threshold_used=self.config.semantic_threshold,
                method="semantic",
                match_rationale="Required rubric criterion or student-answer evidence is blank.",
            )
        # BGE retrieval models use a query instruction for the criterion only.
        # Student-answer sentences are passages and must remain unprefixed.
        query = BGE_QUERY_INSTRUCTION + criterion
        vectors = self.encoder.encode([query, *sentences], normalize_embeddings=True)
        criterion_vector = vectors[0]
        scores = [cosine_similarity(criterion_vector, sentence_vector) for sentence_vector in vectors[1:]]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        score = scores[best_index]
        alias_candidates = [(index, visible_rubric_aliases(criterion, sentence)) for index, sentence in enumerate(sentences)]
        alias_index, aliases = max(alias_candidates, key=lambda item: len(item[1]))
        semantic_match = score >= self.config.semantic_threshold
        alias_match = bool(aliases)
        if alias_match and not semantic_match:
            phrase_index = alias_index
        else:
            phrase_index = best_index
        if semantic_match and alias_match:
            method, rationale = "semantic+visible_rubric_alias", "Embedding threshold and visible-rubric alias both support this criterion."
        elif alias_match:
            method, rationale = "visible_rubric_alias", f"Visible-rubric alias match: {', '.join(aliases)}."
        else:
            method, rationale = "semantic", "Embedding similarity did not reach the configured threshold."
        return SemanticMatchEvidence(
            matched=semantic_match or alias_match,
            similarity_score=round(score, 4),
            matched_answer_phrase=sentences[phrase_index],
            model_name=self.config.model_name,
            threshold_used=self.config.semantic_threshold,
            method=method,
            match_rationale=rationale,
        )
