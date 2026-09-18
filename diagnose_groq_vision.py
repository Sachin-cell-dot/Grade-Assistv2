"""Temporary raw-vision diagnostic; intentionally bypasses GradeAssist schemas."""
import argparse
import base64
import mimetypes
from pathlib import Path
from time import perf_counter

import httpx

from config.settings import get_settings

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

PROMPT = """You are looking at a worksheet image.

Do not solve or grade anything.

Simply describe and transcribe what you can VISUALLY SEE.

Report:

1. Every piece of text visible at the top of the page.
2. Student name, class, subject and date if visible.
3. Every section heading visible.
4. Every question/subquestion label visible.
5. The complete visible question text.
6. The student's written answers exactly as visible.
7. Every red teacher annotation, including ticks, crosses, circles, numeric marks, corrections and comments.
8. Any overall score visible anywhere on the page.

Do not infer missing information.
Do not calculate anything.
Do not return JSON.
Return detailed plain text.

IMPORTANT:
Actually inspect the image. Do not describe what a generic worksheet might contain."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to the local worksheet image to diagnose")
    args = parser.parse_args()
    path = Path(args.image).expanduser().resolve()
    exists = path.is_file()
    print(f"Image path: {path}")
    print(f"Image file exists: {exists}")
    if not exists:
        raise SystemExit("Image path does not point to a file.")

    mime_type, _ = mimetypes.guess_type(path.name)
    image_bytes = path.read_bytes()
    # Re-read through the resolved CLI path to ensure the encoded bytes came from it.
    if image_bytes != path.read_bytes():
        raise RuntimeError("Image changed while being read; diagnostic aborted.")
    if mime_type not in SUPPORTED_IMAGE_TYPES:
        raise SystemExit(f"Unsupported image MIME type: {mime_type}")

    encoded = base64.b64encode(image_bytes).decode("ascii")
    settings = get_settings()
    print(f"Image byte length: {len(image_bytes)}")
    print(f"Detected MIME type: {mime_type}")
    print(f"Base64 length: {len(encoded)}")
    print(f"Model: {settings.groq_vision_model}")
    if not settings.groq_api_key:
        raise SystemExit("GROQ_API_KEY is not configured.")

    payload = {
        "model": settings.groq_vision_model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
        ]}],
        "temperature": 0,
        "max_completion_tokens": 4096,
        "reasoning_effort": "none",
        "reasoning_format": "hidden",
    }
    started = perf_counter()
    try:
        response = httpx.post(
            GROQ_CHAT_COMPLETIONS_URL,
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            json=payload,
            timeout=settings.groq_timeout_seconds,
        )
    except httpx.RequestError as exc:
        print("HTTP status: unavailable")
        print(f"Elapsed time: {perf_counter() - started:.2f}s")
        raise SystemExit(f"Provider request failed: {exc}") from exc

    elapsed = perf_counter() - started
    print(f"HTTP status: {response.status_code}")
    print(f"Elapsed time: {elapsed:.2f}s")
    if response.is_error:
        print(f"Provider response length: {len(response.text)}")
        print(response.text)
        response.raise_for_status()

    body = response.json()
    raw_response = body["choices"][0]["message"]["content"]
    print(f"Provider response length: {len(raw_response)}")
    print(raw_response)


if __name__ == "__main__":
    main()
