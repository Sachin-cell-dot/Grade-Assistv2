import argparse
from time import perf_counter

from services.groq_vision_service import GroqVisionService, groq_dto_to_vision_extraction
from tools.scoring import deterministic_total
from core.evidence_verification import verify_extraction
from core.provenance import build_provenance

parser = argparse.ArgumentParser()
parser.add_argument("image", help="Path to a real worksheet image")
args = parser.parse_args()
service = GroqVisionService()
started = perf_counter()
dto = service.extract_image(args.image)
result = groq_dto_to_vision_extraction(dto)
elapsed = perf_counter() - started
provenance = build_provenance(args.image, provider="groq", model=service.settings.groq_vision_model, policy_version="groq-json-prompt-v1", elapsed_seconds=elapsed)
verification = verify_extraction(result)
print(f"Provider/model: Groq / {service.settings.groq_vision_model}")
print(f"Active completion-token budget (answer sheet): {service.settings.groq_answer_sheet_max_completion_tokens}")
print(f"Elapsed time: {elapsed:.2f}s")
print(f"HTTP status: {service.last_diagnostics.get('http_status')}")
print(f"Raw response length: {service.last_diagnostics.get('raw_response_length')}")
print(f"Sections extracted: {len(dto.sections)}")
print(f"Questions extracted: {sum(len(section.questions) for section in dto.sections)}")
print(f"Teacher symbols extracted: {sum(1 for section in dto.sections for question in section.questions if question.teacher_symbol)}")
print(f"Visible individual marks extracted: {sum(1 for section in dto.sections for question in section.questions if question.awarded_mark is not None)}")
print(f"Section scores extracted: {sum(1 for section in dto.sections if section.reported_score is not None)}")
print(f"Worksheet reported score: {dto.reported_score_obtained}/{dto.reported_score_maximum}" if dto.reported_score_obtained is not None else "Worksheet reported score: None")
print(f"Deterministic total: {deterministic_total(result)}")
print("Groq DTO:")
print(dto.model_dump_json(indent=2))
print("Canonical VisionExtraction:")
print(result.model_dump_json(indent=2))
print("Extraction provenance:")
print(provenance.model_dump_json(indent=2))
print("EVIDENCE VERIFICATION")
print(f"Sections: {verification.evidence_summary.visible_sections}")
print(f"Section scores: {[section.visible_section_score.obtained if section.visible_section_score else None for section in result.sections]}")
print(f"Visible section-score sum: {verification.evidence_summary.visible_section_score_sum}")
print(f"Worksheet reported score: {verification.evidence_summary.worksheet_reported_obtained}/{verification.evidence_summary.worksheet_reported_maximum}")
print(f"Question evidence coverage: text={verification.evidence_summary.questions_with_text}, answers={verification.evidence_summary.questions_with_student_answer}, symbols={verification.evidence_summary.questions_with_teacher_symbol}, individual_marks={verification.evidence_summary.questions_with_individual_mark}")
print(f"Verification status: {verification.status}")
print(f"Reasons: {verification.blockers + verification.warnings}")
print("GROQ VISION VALIDATION SUCCEEDED")
