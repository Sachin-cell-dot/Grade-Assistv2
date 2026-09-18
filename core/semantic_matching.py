"""Local CPU sentence-embedding matching; no provider or LLM calls."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol

from pydantic import Field

from core.models import EvidenceModel

SEMANTIC_METHOD_VERSION = "sentence-transformers-semantic-v1"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


class SentenceEncoder(Protocol):
    def encode(self, sentences: list[str], *, normalize_embeddings: bool) -> object: ...


class SemanticEmbeddingConfig(EvidenceModel):
    model_name: str = DEFAULT_EMBEDDING_MODEL
    semantic_threshold: float = Field(default=0.55, ge=0, le=1)
    allow_lexical_fallback: bool = True


class SemanticMatchEvidence(EvidenceModel):
    matched: bool
    similarity_score: float = Field(ge=0, le=1)
    matched_answer_phrase: str | None = None
    model_name: str
    threshold_used: float = Field(ge=0, le=1)
    method: str = SEMANTIC_METHOD_VERSION


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
    return [sentence.strip() for sentence in _SENTENCE.split(answer) if sentence.strip()] or [answer.strip()]


class SemanticSentenceMatcher:
    def __init__(self, config: SemanticEmbeddingConfig | None = None, encoder: SentenceEncoder | None = None):
        self.config = config or SemanticEmbeddingConfig()
        self.encoder = encoder or load_sentence_encoder(self.config.model_name)

    def match(self, criterion: str, answer: str) -> SemanticMatchEvidence:
        sentences = answer_sentences(answer)
        vectors = self.encoder.encode([criterion, *sentences], normalize_embeddings=True)
        criterion_vector = vectors[0]
        scores = [max(0.0, min(1.0, float(sum(left * right for left, right in zip(criterion_vector, sentence_vector))))) for sentence_vector in vectors[1:]]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        score = scores[best_index]
        return SemanticMatchEvidence(
            matched=score >= self.config.semantic_threshold,
            similarity_score=round(score, 4),
            matched_answer_phrase=sentences[best_index],
            model_name=self.config.model_name,
            threshold_used=self.config.semantic_threshold,
        )
