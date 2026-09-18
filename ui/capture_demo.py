from pathlib import Path
import tempfile
import streamlit as st
from config.settings import get_settings
from tools.vision import CaptureConfig, extract_candidate_frames


def render_capture_demo() -> None:
    st.subheader("Video capture preview")
    st.caption("Local OpenCV frame selection only — no AI extraction is run here.")
    upload = st.file_uploader("Upload a teacher-marking video", type=["mp4", "avi", "mov"], key="capture_video")
    if not upload:
        st.info("Upload a short video to select stable, sharp worksheet frames.")
        return
    settings = get_settings()
    config = CaptureConfig(
        settings.capture_sample_every_n_frames, settings.capture_page_change_threshold,
        settings.capture_stability_threshold, settings.capture_stable_frames_required,
        settings.capture_blur_threshold, settings.capture_min_frames_between_saves,
    )
    st.video(upload)
    if st.button("Select candidate frames", type="primary"):
        with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix, delete=False) as temporary:
            temporary.write(upload.getvalue())
            source = Path(temporary.name)
        output = Path("data/processed/candidates")
        try:
            with st.spinner("Checking page changes, stability, and blur..."):
                summary = extract_candidate_frames(source, output, config)
            st.success(f"Selected {len(summary.candidates)} candidate frame(s).")
            st.json({"frames_read": summary.frames_read, "sampled_frames": summary.sampled_frames, "page_changes": summary.page_changes, "blurry_frames_rejected": summary.blurry_frames_rejected})
            for candidate in summary.candidates:
                st.image(str(candidate.path), caption=f"Frame {candidate.frame_index} at {candidate.timestamp_seconds:.2f}s — blur variance {candidate.blur_variance:.1f}")
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
