from pathlib import Path

import tools.seed_verified_document as seed
from services.groq_vision_service import GroqVisionError, GroqQuestionPaperExtraction


def test_seed_one_question_paper_writes_raw_json_and_imports_cache(monkeypatch, tmp_path, capsys):
    image = tmp_path / "paper.jpeg"
    image.write_bytes(b"image")
    monkeypatch.chdir(tmp_path)

    class Settings:
        groq_vision_model = "qwen/test"

    class FakeService:
        settings = Settings()
        def extract_question_paper(self, path):
            return GroqQuestionPaperExtraction.model_validate({"worksheet_title": "Paper", "sections": [{"questions": [{"identifier": "1", "printed_text": "Question"}]}]})

    monkeypatch.setattr(seed, "GroqVisionService", FakeService)
    assert seed.main(["seed_verified_document.py", "question_paper", str(image)]) == 0
    assert (tmp_path / "data/processed/paper_live.json").is_file()
    assert "Verified local result" in capsys.readouterr().out


def test_seed_stops_after_single_rate_limit_error(monkeypatch, tmp_path, capsys):
    image = tmp_path / "paper.jpeg"
    image.write_bytes(b"image")

    class FakeService:
        def extract_question_paper(self, path):
            raise GroqVisionError("Groq Vision returned HTTP 429.")

    monkeypatch.setattr(seed, "GroqVisionService", FakeService)
    assert seed.main(["seed_verified_document.py", "question_paper", str(image)]) == 1
    assert capsys.readouterr().err == "Groq is temporarily rate-limited. Please wait briefly, then retry.\n"
