from pathlib import Path

import ui.extraction_demo as extraction_demo
from services.groq_vision_service import GroqWorksheetExtraction


def test_uploaded_answer_sheet_uses_groq_and_never_imports_ollama(monkeypatch, tmp_path):
    image = tmp_path / "rufina.jpeg"
    image.write_bytes(b"private-image-bytes")
    called = []

    class FakeSettings:
        groq_vision_model = "qwen/qwen3.8-27b"

    class FakeGroqService:
        settings = FakeSettings()

        def extract_answer_sheet(self, image_path):
            called.append(Path(image_path))
            return GroqWorksheetExtraction.model_validate({
                "student_name": "Rufina",
                "sections": [{"identifier": "Part A", "questions": [{"identifier": "Q1", "student_answer": "Visible answer", "awarded_mark": 2}]}],
                "reported_score_obtained": 2,
                "reported_score_maximum": 20,
            })

    monkeypatch.setattr(extraction_demo, "GroqVisionService", FakeGroqService)
    extraction, provenance = extraction_demo.extract_uploaded_answer_sheet(image)

    assert called == [image]
    assert extraction.student.name == "Rufina"
    assert extraction.sections[0].questions[0].student_answer.visible_text == "Visible answer"
    assert provenance.provider == "groq"
    assert provenance.model == "qwen/qwen3.8-27b"
    assert "OllamaVisionService" not in extraction_demo.__dict__
