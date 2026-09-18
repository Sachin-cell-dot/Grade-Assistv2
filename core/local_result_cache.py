"""Ignored, local-only cache for successful Groq demo results."""
from hashlib import sha256
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from core.provenance import ExtractionProvenance

DEFAULT_CACHE_DIRECTORY = Path("data/processed/groq_cache")


class CachedGroqResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cache_key: str
    image_fingerprint_sha256: str
    document_type: str
    model: str
    policy_version: str
    result_payload: dict
    provenance: ExtractionProvenance


def image_fingerprint(image_path: str | Path) -> str:
    return sha256(Path(image_path).read_bytes()).hexdigest()


def cache_key_for(image_path: str | Path, document_type: str, model: str, policy_version: str) -> tuple[str, str]:
    fingerprint = image_fingerprint(image_path)
    material = "|".join((fingerprint, document_type, model, policy_version))
    return sha256(material.encode("utf-8")).hexdigest(), fingerprint


class LocalGroqResultCache:
    def __init__(self, directory: str | Path = DEFAULT_CACHE_DIRECTORY):
        self.directory = Path(directory)

    def get(self, image_path: str | Path, document_type: str, model: str, policy_version: str) -> CachedGroqResult | None:
        key, _ = cache_key_for(image_path, document_type, model, policy_version)
        location = self.directory / f"{key}.json"
        if not location.is_file():
            return None
        try:
            cached = CachedGroqResult.model_validate_json(location.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return cached if cached.cache_key == key else None

    def put(self, image_path: str | Path, document_type: str, model: str, policy_version: str, result_payload: dict, provenance: ExtractionProvenance) -> CachedGroqResult:
        key, fingerprint = cache_key_for(image_path, document_type, model, policy_version)
        cached = CachedGroqResult(
            cache_key=key,
            image_fingerprint_sha256=fingerprint,
            document_type=document_type,
            model=model,
            policy_version=policy_version,
            result_payload=result_payload,
            provenance=provenance,
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / f"{key}.json").write_text(cached.model_dump_json(indent=2), encoding="utf-8")
        return cached

    def latest_for_document_type(self, document_type: str, model: str, policy_version: str) -> CachedGroqResult | None:
        """Return the newest successful local result for an explicitly requested document type."""
        if not self.directory.is_dir():
            return None
        candidates: list[CachedGroqResult] = []
        for location in self.directory.glob("*.json"):
            try:
                cached = CachedGroqResult.model_validate_json(location.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if cached.document_type == document_type and cached.model == model and cached.policy_version == policy_version:
                candidates.append(cached)
        return max(candidates, key=lambda item: item.provenance.extracted_at) if candidates else None
