import base64, json
from pathlib import Path
import httpx
from config.settings import Settings, get_settings
from core.logger import get_logger
from core.models import VisionExtraction

PROMPT = """You are a worksheet EVIDENCE TRANSCRIBER, not a grader. Output JSON only using the supplied schema.

Inspect the ENTIRE page top-to-bottom. Transcribe both printed/student-written content and teacher-written annotations, including red ink. Teacher handwriting is first-class worksheet evidence: a visibly handwritten score such as 14/20 is valid reported-score evidence with obtained=14 and maximum=20; red section/topic scores are section scores; red numeric marks beside questions are individual awarded marks; ticks/crosses/circles are teacher markings; comments such as “Should Improve” are teacher comments. Do not require evidence to appear in a predefined printed field.

Scan TOP for student metadata. Scan BODY for every visible section/title, every visible question/subquestion, complete question text, student answer, visible working, teacher markings/corrections, individual marks, and section scores. Scan BOTTOM and MARGINS for overall teacher scores, comments, and unassigned red annotations. Create a section/question whenever it is visible even if a field inside it is unreadable; use null and field uncertainty only for that missing evidence. Never return sections=[] when visible worksheet sections or questions can be read.

Associate a small annotation with a question only when visibly adjacent; otherwise preserve it as an unassigned teacher marking. Never solve questions, correct a student answer, judge correctness, derive marks, infer marks from a total, fabricate markings, or derive section scores. Do not invent identifiers. Do not calculate totals."""

class OllamaVisionError(RuntimeError): pass

class OllamaVisionService:
    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None):
        self.settings = settings or get_settings()
        self.client = client or httpx.Client(timeout=self.settings.ollama_timeout_seconds)
        self.logger = get_logger(__name__, self.settings.log_level)
    def extract_image(self, image_path: str | Path) -> VisionExtraction:
        path = Path(image_path)
        if not path.is_file(): raise OllamaVisionError("Worksheet image was not found.")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        payload = {"model": self.settings.ollama_vision_model, "prompt": PROMPT, "images": [encoded], "stream": False, "format": VisionExtraction.model_json_schema(), "options": {"num_predict": self.settings.ollama_num_predict, "num_ctx": self.settings.ollama_num_ctx}}
        try:
            response = self.client.post(f"{self.settings.ollama_host.rstrip('/')}/api/generate", json=payload); response.raise_for_status()
        except httpx.TimeoutException as exc: raise OllamaVisionError("Ollama request timed out; increase OLLAMA_TIMEOUT_SECONDS or use a smaller image.") from exc
        except httpx.ConnectError as exc: raise OllamaVisionError("Ollama is unavailable. Start Ollama and verify OLLAMA_HOST.") from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300].replace("\n", " ")
            raise OllamaVisionError(f"Ollama returned HTTP {exc.response.status_code}. {detail}") from exc
        try:
            body = response.json()
            if body.get("done") is False: raise OllamaVisionError("Ollama output was incomplete or truncated.")
            raw = body.get("response")
            if not isinstance(raw, str) or not raw.strip(): raise OllamaVisionError("Ollama returned no structured response.")
            return VisionExtraction.model_validate(json.loads(raw))
        except json.JSONDecodeError as exc: raise OllamaVisionError("Ollama returned malformed JSON, not a valid extraction.") from exc
        except ValueError as exc: raise OllamaVisionError(f"Ollama JSON failed the extraction schema: {exc}") from exc
