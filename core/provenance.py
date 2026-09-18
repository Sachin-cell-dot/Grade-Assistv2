"""Non-secret extraction-run provenance kept separate from worksheet evidence."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ExtractionProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str
    model: str
    extracted_at: datetime
    source_image_identifier: str
    source_image_fingerprint_sha256: str
    policy_version: str
    elapsed_seconds: float | None = None


def build_provenance(
    image_path: str | Path,
    *,
    provider: str,
    model: str,
    policy_version: str,
    elapsed_seconds: float | None = None,
) -> ExtractionProvenance:
    path = Path(image_path)
    return ExtractionProvenance(
        provider=provider,
        model=model,
        extracted_at=datetime.now(timezone.utc),
        source_image_identifier=path.name,
        source_image_fingerprint_sha256=sha256(path.read_bytes()).hexdigest(),
        policy_version=policy_version,
        elapsed_seconds=elapsed_seconds,
    )
