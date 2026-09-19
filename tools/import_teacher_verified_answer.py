"""Import one canonical, teacher-verified answer-sheet JSON without providers."""
from __future__ import annotations

import argparse

from core.teacher_evidence_import import (
    TEACHER_VERIFIED_LABEL,
    TeacherEvidenceImportError,
    import_teacher_verified_answer_evidence,
)
from ui.extraction_demo import POLICY_VERSION


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    parser.add_argument("--source-image", required=True, help="Original worksheet image filename only")
    parser.add_argument("--note")
    args = parser.parse_args()
    try:
        cached = import_teacher_verified_answer_evidence(
            args.json_file, source_image_filename=args.source_image, note=args.note, policy_version=POLICY_VERSION,
        )
    except TeacherEvidenceImportError as exc:
        parser.error(str(exc))
    print(TEACHER_VERIFIED_LABEL)
    print(f"CACHE_KEY={cached.cache_key}")
    print(f"SOURCE_IMAGE={cached.provenance.source_image_identifier}")
    print(f"SOURCE_TYPE={cached.provenance.source_type}")
    print(f"POLICY={cached.policy_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
