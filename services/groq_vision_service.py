"""Experimental Groq Vision provider; it does not replace the Ollama provider."""
import base64
import json
import mimetypes
from pathlib import Path
from time import perf_counter

import httpx
from pydantic import BaseModel, ConfigDict, Field

from config.settings import Settings, get_settings
from core.logger import get_logger
from core.models import (
    Question,
    Section,
    StudentAnswer,
    StudentMetadata,
    TeacherMarking,
    Uncertainty,
    VisibleScore,
    VisionExtraction,
)

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
PROMPT = """You are a visual worksheet evidence transcriber.

Actually inspect the supplied image from top to bottom. Return ONLY one valid JSON object. No Markdown. No code fences. No explanation before or after JSON.

Use exactly this shape:
{
  "student_name": string|null,
  "roll_number": string|null,
  "class_name": string|null,
  "subject": string|null,
  "assessment_date": string|null,
  "worksheet_title": string|null,
  "sections": [{"identifier": string|null, "title": string|null, "reported_score": number|null, "questions": [{"identifier": string|null, "question_text": string|null, "student_answer": string|null, "visible_working": string|null, "teacher_symbol": string|null, "teacher_correction": string|null, "awarded_mark": number|null, "teacher_comment": string|null}]}],
  "reported_score_obtained": number|null,
  "reported_score_maximum": number|null,
  "teacher_comment": string|null,
  "unassigned_teacher_annotations": [string]
}

IMPORTANT: Do not return an empty object merely because some fields are unreadable. If visible worksheet sections/questions exist, transcribe them. Printed text, student handwriting and teacher red pen are ALL evidence. A handwritten teacher score such as 14/20 is valid visible evidence. A red section-margin score is section-level evidence. A red tick/cross is teacher-symbol evidence. Preserve student answers exactly, including wrong answers.

Do NOT solve, grade, determine correctness, infer marks, convert ticks/crosses to numeric marks, calculate totals, or invent identifiers. If one field is unreadable, use null for THAT FIELD while preserving the visible parent section/question."""


class GroqDto(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GroqQuestion(GroqDto):
    identifier: str | None = None
    question_text: str | None = None
    student_answer: str | None = None
    visible_working: str | None = None
    teacher_symbol: str | None = None
    teacher_correction: str | None = None
    awarded_mark: float | None = Field(default=None, ge=0)
    teacher_comment: str | None = None


class GroqSection(GroqDto):
    identifier: str | None = None
    title: str | None = None
    reported_score: float | None = Field(default=None, ge=0)
    questions: list[GroqQuestion] = Field(default_factory=list)


class GroqWorksheetExtraction(GroqDto):
    student_name: str | None = None
    roll_number: str | None = None
    class_name: str | None = None
    subject: str | None = None
    assessment_date: str | None = None
    worksheet_title: str | None = None
    sections: list[GroqSection] = Field(default_factory=list)
    reported_score_obtained: float | None = Field(default=None, ge=0)
    reported_score_maximum: float | None = Field(default=None, ge=0)
    teacher_comment: str | None = None
    unassigned_teacher_annotations: list[str] = Field(default_factory=list)


def _absence_uncertainty() -> Uncertainty:
    return Uncertainty(is_uncertain=True, source_legibility="not_visible")


def _teacher_marking(question: GroqQuestion) -> TeacherMarking | None:
    if not any((question.teacher_symbol, question.teacher_correction, question.awarded_mark is not None, question.teacher_comment)):
        return None
    return TeacherMarking(
        visible_symbols=[question.teacher_symbol] if question.teacher_symbol else [],
        visible_correction=question.teacher_correction,
        visible_comment=question.teacher_comment,
        visible_individual_score=VisibleScore(obtained=question.awarded_mark) if question.awarded_mark is not None else None,
    )


def groq_dto_to_vision_extraction(dto: GroqWorksheetExtraction) -> VisionExtraction:
    """Pure, deterministic conversion from the small provider DTO to the canonical domain model."""
    sections = []
    for dto_section in dto.sections:
        questions = []
        for dto_question in dto_section.questions:
            answer_present = dto_question.student_answer is not None or dto_question.visible_working is not None
            questions.append(Question(
                identifier=dto_question.identifier,
                question_text=dto_question.question_text,
                student_answer=StudentAnswer(
                    visible_text=dto_question.student_answer,
                    visible_working=dto_question.visible_working,
                    uncertainty=Uncertainty() if answer_present else _absence_uncertainty(),
                ),
                teacher_marking=_teacher_marking(dto_question),
            ))
        section_present = any((dto_section.identifier, dto_section.title, dto_section.reported_score is not None, questions))
        sections.append(Section(
            identifier=dto_section.identifier,
            title=dto_section.title,
            visible_section_score=VisibleScore(obtained=dto_section.reported_score) if dto_section.reported_score is not None else None,
            questions=questions,
            uncertainty=Uncertainty() if section_present else _absence_uncertainty(),
        ))
    student_values = (dto.student_name, dto.roll_number, dto.class_name, dto.subject, dto.assessment_date, dto.worksheet_title)
    annotations = list(dto.unassigned_teacher_annotations)
    if dto.teacher_comment is not None:
        annotations.append(dto.teacher_comment)
    has_any_evidence = any(value is not None for value in student_values) or bool(sections) or bool(annotations) or dto.reported_score_obtained is not None
    return VisionExtraction(
        student=StudentMetadata(
            name=dto.student_name,
            roll_number=dto.roll_number,
            class_name=dto.class_name,
            subject=dto.subject,
            assessment_date=dto.assessment_date,
            uncertainty=Uncertainty() if any(value is not None for value in student_values) else _absence_uncertainty(),
        ),
        sections=sections,
        worksheet_reported_score=VisibleScore(obtained=dto.reported_score_obtained, maximum=dto.reported_score_maximum)
        if dto.reported_score_obtained is not None else None,
        unassigned_teacher_markings=[TeacherMarking(visible_comment=annotation) for annotation in annotations],
        worksheet_score_uncertainty=Uncertainty() if dto.reported_score_obtained is not None else _absence_uncertainty(),
        overall_uncertainty=Uncertainty() if has_any_evidence else _absence_uncertainty(),
    )


class GroqVisionError(RuntimeError):
    pass


class GroqVisionExtractionError(GroqVisionError):
    pass


class GroqVisionService:
    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None):
        self.settings = settings or get_settings()
        self.client = client or httpx.Client(timeout=self.settings.groq_timeout_seconds)
        self.logger = get_logger(__name__, self.settings.log_level)
        self.last_diagnostics: dict[str, int | float] = {}

    @staticmethod
    def _strip_json_fence(raw: str) -> str:
        stripped = raw.strip()
        if not stripped.startswith("```"):
            return stripped
        first_newline = stripped.find("\n")
        if first_newline == -1 or not stripped.endswith("```"):
            return stripped
        header = stripped[:first_newline].strip().lower()
        if header not in {"```", "```json"}:
            return stripped
        return stripped[first_newline + 1:-3].strip()

    @staticmethod
    def _is_semantically_empty(dto: GroqWorksheetExtraction) -> bool:
        return (
            all(value is None for value in (
                dto.student_name, dto.roll_number, dto.class_name, dto.subject,
                dto.assessment_date, dto.worksheet_title, dto.reported_score_obtained,
                dto.reported_score_maximum, dto.teacher_comment,
            ))
            and not dto.sections
            and not dto.unassigned_teacher_annotations
        )

    def _image_data_url(self, image_path: str | Path) -> str:
        path = Path(image_path)
        if not path.is_file():
            raise GroqVisionError("Worksheet image was not found.")
        content_type, _ = mimetypes.guess_type(path.name)
        if content_type not in SUPPORTED_IMAGE_TYPES:
            raise GroqVisionError("Groq Vision supports JPEG, PNG, and WebP worksheet images only.")
        size = path.stat().st_size
        if size > self.settings.groq_max_image_bytes:
            raise GroqVisionError(f"Worksheet image is too large ({size} bytes); limit is {self.settings.groq_max_image_bytes} bytes.")
        return f"data:{content_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"

    def extract_image(self, image_path: str | Path) -> GroqWorksheetExtraction:
        if not self.settings.groq_api_key:
            raise GroqVisionError("GROQ_API_KEY is not configured.")
        image_url = self._image_data_url(image_path)
        payload = {
            "model": self.settings.groq_vision_model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]}],
            "temperature": 0,
            "max_completion_tokens": 4096,
            "reasoning_effort": "none",
            "reasoning_format": "hidden",
        }
        started = perf_counter()
        try:
            response = self.client.post(
                GROQ_CHAT_COMPLETIONS_URL,
                headers={"Authorization": f"Bearer {self.settings.groq_api_key}"},
                json=payload,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise GroqVisionError("Groq Vision request timed out.") from exc
        except httpx.RequestError as exc:
            raise GroqVisionError("Groq Vision request failed; check network access and GROQ_API_KEY.") from exc
        except httpx.HTTPStatusError as exc:
            raise GroqVisionError(f"Groq Vision returned HTTP {exc.response.status_code}.") from exc
        try:
            body = response.json()
            raw = body["choices"][0]["message"]["content"]
            if not isinstance(raw, str) or not raw.strip():
                raise GroqVisionError("Groq Vision returned no JSON content.")
            self.last_diagnostics = {
                "http_status": response.status_code,
                "elapsed_seconds": perf_counter() - started,
                "raw_response_length": len(raw),
            }
            dto = GroqWorksheetExtraction.model_validate(json.loads(self._strip_json_fence(raw)))
            if self._is_semantically_empty(dto):
                raise GroqVisionExtractionError("Provider returned structurally valid but semantically empty extraction.")
            return dto
        except (KeyError, IndexError, TypeError) as exc:
            raise GroqVisionError("Groq Vision returned an unexpected response shape.") from exc
        except json.JSONDecodeError as exc:
            raise GroqVisionError("Groq Vision returned malformed JSON.") from exc
        except ValueError as exc:
            raise GroqVisionError(f"Groq Vision JSON failed the Groq DTO schema: {exc}") from exc
