"""Seed one locally verified Groq document result, one explicit invocation."""
from __future__ import annotations

import sys
from pathlib import Path

from core.rubric_import import VerifiedDocumentImportError, import_verified_live_document_json
from services.groq_vision_service import GroqVisionError, GroqVisionService
from ui.extraction_demo import POLICY_VERSION

USAGE = "Usage: py -3.11 -u tools\\seed_verified_document.py {question_paper|rubric|answer_sheet} <image path>"


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in {"question_paper", "rubric", "answer_sheet"}:
        print(USAGE, file=sys.stderr)
        return 2
    kind, image_text = argv[1], argv[2]
    image_path = Path(image_text)
    if not image_path.is_file():
        print(f"Input image was not found: {image_path}", file=sys.stderr)
        return 2
    service = GroqVisionService()
    try:
        if kind == "question_paper":
            dto = service.extract_question_paper(image_path)
        elif kind == "rubric":
            dto = service.extract_rubric(image_path)
        else:
            dto = service.extract_answer_sheet(image_path)
    except GroqVisionError as exc:
        if "HTTP 429" in str(exc):
            print("Groq is temporarily rate-limited. Please wait briefly, then retry.", file=sys.stderr)
        else:
            print(f"Groq extraction failed: {exc}", file=sys.stderr)
        return 1
    raw_path = Path("data/processed") / f"{image_path.stem.lower()}_live.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(dto.model_dump_json(indent=2), encoding="utf-8")
    try:
        cached = import_verified_live_document_json(raw_path, kind, model=service.settings.groq_vision_model, policy_version=POLICY_VERSION)
    except VerifiedDocumentImportError as exc:
        print(f"Typed cache import failed: {exc}", file=sys.stderr)
        return 1
    print("Verified local result")
    print(f"CACHE_KEY={cached.cache_key}")
    print(f"DOCUMENT_TYPE={cached.document_type}")
    print(f"SOURCE={cached.provenance.source_image_identifier}")
    print(f"PROVIDER={cached.provenance.provider}")
    print(f"MODEL={cached.model}")
    print(f"POLICY={cached.policy_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
