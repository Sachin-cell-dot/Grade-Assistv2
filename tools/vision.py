from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4
import cv2
import numpy as np

def validate_worksheet_image(path: str | Path) -> tuple[bool, str]:
    image = cv2.imread(str(path))
    if image is None:
        return False, "OpenCV could not read this worksheet image."
    return True, f"Image validated: {image.shape[1]} x {image.shape[0]} pixels"


def list_candidate_images(directory: str | Path = "data/processed/candidates") -> list[Path]:
    """Return locally selected frame images, newest first."""
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


@dataclass(frozen=True)
class CaptureConfig:
    sample_every_n_frames: int = 5
    page_change_threshold: float = 18.0
    stability_threshold: float = 3.0
    stable_frames_required: int = 3
    blur_threshold: float = 80.0
    min_frames_between_saves: int = 30


@dataclass(frozen=True)
class FrameCandidate:
    frame_index: int
    timestamp_seconds: float
    blur_variance: float
    path: Path


@dataclass
class CaptureSummary:
    frames_read: int = 0
    sampled_frames: int = 0
    page_changes: int = 0
    blurry_frames_rejected: int = 0
    candidates: list[FrameCandidate] = field(default_factory=list)


def _preview_gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (320, 240), interpolation=cv2.INTER_AREA)


def blur_variance(frame: np.ndarray) -> float:
    """Laplacian variance: lower values indicate a likely blurry frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_candidate_frames(video_path: str | Path, output_dir: str | Path, config: CaptureConfig) -> CaptureSummary:
    """Select stable, sharp page frames locally; no OCR or AI is performed here."""
    source = Path(video_path)
    if not source.is_file():
        raise ValueError("Video file was not found.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoCapture(str(source))
    if not video.isOpened():
        raise ValueError("OpenCV could not open this video.")
    fps = video.get(cv2.CAP_PROP_FPS) or 1.0
    summary = CaptureSummary()
    previous_preview: np.ndarray | None = None
    stable_count = 0
    last_saved_frame = -config.min_frames_between_saves
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            summary.frames_read += 1
            frame_index = summary.frames_read - 1
            if frame_index % config.sample_every_n_frames:
                continue
            summary.sampled_frames += 1
            preview = _preview_gray(frame)
            if previous_preview is None:
                stable_count = 1
            else:
                difference = float(cv2.absdiff(preview, previous_preview).mean())
                if difference >= config.page_change_threshold:
                    summary.page_changes += 1
                    stable_count = 1
                elif difference <= config.stability_threshold:
                    stable_count += 1
                else:
                    stable_count = 1
            previous_preview = preview
            if stable_count < config.stable_frames_required or frame_index - last_saved_frame < config.min_frames_between_saves:
                continue
            sharpness = blur_variance(frame)
            if sharpness < config.blur_threshold:
                summary.blurry_frames_rejected += 1
                continue
            filename = f"candidate_{frame_index:06d}_{uuid4().hex[:8]}.jpg"
            saved_path = destination / filename
            if not cv2.imwrite(str(saved_path), frame):
                raise RuntimeError("OpenCV could not save a selected frame.")
            summary.candidates.append(FrameCandidate(frame_index, frame_index / fps, sharpness, saved_path))
            last_saved_frame = frame_index
            stable_count = 0
    finally:
        video.release()
    return summary
