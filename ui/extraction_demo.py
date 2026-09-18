from pathlib import Path
import tempfile
import streamlit as st
from core.models import VisionExtraction
from services.ollama_service import OllamaVisionError, OllamaVisionService
from tools.scoring import deterministic_total
from tools.vision import list_candidate_images, validate_worksheet_image


def _choose_image() -> tuple[Path | None, str | None]:
    candidates = list_candidate_images()
    source = st.radio("Image source", ["Fictional demo worksheet", "Upload image", "Selected capture frame"], horizontal=True)
    if source == "Fictional demo worksheet":
        demo = Path("data/demo/fictional_marked_worksheet.png")
        if not demo.is_file():
            st.error("The fictional demo worksheet is unavailable.")
            return None, None
        return demo, "Fictional demo worksheet — no real student data"
    if source == "Selected capture frame":
        if not candidates:
            st.info("No saved capture candidates yet. Use the Capture frames tab or upload an image.")
            return None, None
        selected = st.selectbox("Candidate frame", candidates, format_func=lambda path: path.name)
        return selected, f"Selected capture frame: {selected.name}"
    upload = st.file_uploader("Upload a worksheet image", type=["png", "jpg", "jpeg", "webp"])
    if not upload:
        return None, None
    destination = Path("data/uploads")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix or ".png", dir=destination, delete=False) as temp:
        temp.write(upload.getvalue())
        return Path(temp.name), f"Uploaded image: {upload.name}"


def render_extraction_demo() -> None:
    st.subheader("Extraction preview — teacher review required")
    st.caption("Visible evidence only. GradeAssist does not solve, grade, or alter the worksheet.")
    image_path, label = _choose_image()
    if image_path is None:
        st.info("Choose a local candidate frame or upload a worksheet image to run a manual extraction.")
        return
    ok, message = validate_worksheet_image(image_path)
    if not ok:
        st.error(message)
        return
    st.success(message)
    st.image(str(image_path), caption=label)
    if st.button("Run evidence extraction", type="primary"):
        try:
            st.session_state["latest_extraction"] = OllamaVisionService().extract_image(image_path).model_dump(mode="json")
        except OllamaVisionError as exc:
            st.error(str(exc))
            return
    result_json = st.session_state.get("latest_extraction")
    if result_json:
        result = VisionExtraction.model_validate(result_json)
        st.json(result_json)
        left, right = st.columns(2)
        left.metric("Deterministic total (visible individual marks)", deterministic_total(result))
        right.metric("Worksheet-reported total (as extracted)", result.worksheet_reported_score if result.worksheet_reported_score is not None else "Not visible")
