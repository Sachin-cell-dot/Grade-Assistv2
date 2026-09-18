import cv2
import fitz
import numpy as np
import pytest

from tools.vision import validate_worksheet_image
from tools.worksheet_input import WorksheetInputError, prepare_worksheet_upload


def _pdf_bytes(page_count: int) -> bytes:
    document = fitz.open()
    for number in range(page_count):
        page = document.new_page()
        page.insert_text((72, 72), f"Worksheet page {number + 1}")
    content = document.tobytes()
    document.close()
    return content


def test_jpeg_and_png_uploads_are_accepted_as_images(tmp_path):
    image = np.full((20, 20, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    prepared = prepare_worksheet_upload("english.jpeg", encoded.tobytes(), tmp_path)
    assert prepared.source_path == prepared.image_path
    assert prepared.image_path.suffix == ".jpeg"
    assert validate_worksheet_image(prepared.image_path)[0]


def test_one_page_pdf_is_rendered_locally_to_a_valid_image(tmp_path):
    prepared = prepare_worksheet_upload("english.pdf", _pdf_bytes(1), tmp_path)
    assert prepared.source_path.suffix == ".pdf"
    assert prepared.image_path.suffix == ".png"
    assert prepared.image_path.is_file()
    assert validate_worksheet_image(prepared.image_path)[0]
    assert "rendered locally" in prepared.label


def test_multi_page_pdf_and_invalid_files_are_rejected(tmp_path):
    with pytest.raises(WorksheetInputError, match="one worksheet page at a time"):
        prepare_worksheet_upload("two-pages.pdf", _pdf_bytes(2), tmp_path)
    with pytest.raises(WorksheetInputError, match="Supported formats"):
        prepare_worksheet_upload("worksheet.docx", b"not a worksheet", tmp_path)
    with pytest.raises(WorksheetInputError, match="empty"):
        prepare_worksheet_upload("worksheet.png", b"", tmp_path)
