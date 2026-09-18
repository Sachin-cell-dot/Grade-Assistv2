"""Local-only worksheet upload preparation for the Streamlit demo."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

ACCEPTED_EXTENSIONS = {".jpeg", ".jpg", ".png", ".pdf"}


class WorksheetInputError(ValueError):
    pass


@dataclass(frozen=True)
class PreparedWorksheet:
    source_path: Path
    image_path: Path
    label: str


def _safe_destination(filename: str, content: bytes, directory: Path) -> Path:
    suffix = Path(filename).suffix.lower()
    if suffix not in ACCEPTED_EXTENSIONS:
        raise WorksheetInputError("Unsupported file. Supported formats: JPEG, PNG, PDF.")
    if not content:
        raise WorksheetInputError("The uploaded file is empty.")
    digest = sha256(content).hexdigest()[:16]
    stem = Path(filename).stem or "worksheet"
    destination = directory / f"{stem}-{digest}{suffix}"
    if not destination.exists():
        destination.write_bytes(content)
    return destination


def _render_single_page_pdf(pdf_path: Path) -> Path:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise WorksheetInputError("PDF support is unavailable locally. Install the configured PyMuPDF dependency.") from exc
    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        raise WorksheetInputError("The uploaded PDF could not be opened locally.") from exc
    try:
        if document.page_count != 1:
            raise WorksheetInputError("This demo currently processes one worksheet page at a time. Upload a one-page PDF.")
        rendered = pdf_path.with_name(f"{pdf_path.stem}-page-1.png")
        if not rendered.exists():
            page = document.load_page(0)
            page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(rendered)
        return rendered
    finally:
        document.close()


def prepare_worksheet_upload(filename: str, content: bytes, upload_directory: str | Path = "data/uploads") -> PreparedWorksheet:
    """Persist an accepted local upload and return an image suitable for OpenCV/vision.

    PDFs are rendered locally; no source file is sent to a conversion service.
    """
    directory = Path(upload_directory)
    directory.mkdir(parents=True, exist_ok=True)
    source_path = _safe_destination(filename, content, directory)
    if source_path.suffix.lower() == ".pdf":
        image_path = _render_single_page_pdf(source_path)
        return PreparedWorksheet(source_path=source_path, image_path=image_path, label=f"Uploaded PDF, rendered locally: {Path(filename).name}")
    return PreparedWorksheet(source_path=source_path, image_path=source_path, label=f"Uploaded image: {Path(filename).name}")
