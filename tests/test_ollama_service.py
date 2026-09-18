import json
import httpx
from config.settings import Settings
from services.ollama_service import OllamaVisionError, OllamaVisionService

def make_service(handler):
    return OllamaVisionService(Settings(ollama_host="http://test"), httpx.Client(transport=httpx.MockTransport(handler)))

def test_extracts_valid_mocked_response(tmp_path):
    image = tmp_path / "paper.jpg"; image.write_bytes(b"image")
    payload = {"student": {"name": "Demo Student"}, "sections": [], "worksheet_reported_score": {"obtained": 8}}
    service = make_service(lambda request: httpx.Response(200, json={"done": True, "response": json.dumps(payload)}))
    assert service.extract_image(image).student.name == "Demo Student"

def test_extracts_full_visible_evidence_without_forced_identifier(tmp_path):
    image = tmp_path / "paper.jpg"; image.write_bytes(b"image")
    payload = {"student": {"name": "Demo Student"}, "sections": [{"identifier": "A", "title": "Arithmetic", "visible_section_score": {"obtained": 4}, "questions": [{"identifier": None, "question_text": "7 + 5 =", "student_answer": {"visible_text": "12", "visible_working": "7 + 5"}, "teacher_marking": {"visible_symbols": ["tick"], "visible_individual_score": {"obtained": 2}}}]}], "worksheet_reported_score": {"obtained": 4}, "unassigned_teacher_markings": [{"visible_comment": "Check presentation", "uncertainty": {"is_uncertain": True, "reasons": ["not adjacent to a question"], "source_legibility": "unclear"}}]}
    service = make_service(lambda request: httpx.Response(200, json={"done": True, "response": json.dumps(payload)}))
    result = service.extract_image(image)
    assert result.sections[0].questions[0].identifier is None
    assert result.sections[0].questions[0].student_answer.visible_working == "7 + 5"
    assert result.unassigned_teacher_markings[0].visible_comment == "Check presentation"

def test_rejects_malformed_json(tmp_path):
    image = tmp_path / "paper.jpg"; image.write_bytes(b"image")
    service = make_service(lambda request: httpx.Response(200, json={"done": True, "response": "not json"}))
    try: service.extract_image(image)
    except OllamaVisionError as exc: assert "malformed JSON" in str(exc)
    else: raise AssertionError("Expected OllamaVisionError")
