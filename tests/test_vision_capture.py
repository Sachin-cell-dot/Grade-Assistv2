import cv2
import numpy as np
from tools.vision import CaptureConfig, blur_variance, extract_candidate_frames


def test_blur_variance_prefers_sharp_image():
    sharp = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.putText(sharp, "MARK", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    blurry = cv2.GaussianBlur(sharp, (21, 21), 0)
    assert blur_variance(sharp) > blur_variance(blurry)


def test_selects_a_stable_sharp_frame(tmp_path):
    video_path = tmp_path / "teacher.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(frame, (20, 20), (140, 100), (255, 255, 255), 3)
    for _ in range(20): writer.write(frame)
    writer.release()
    summary = extract_candidate_frames(video_path, tmp_path / "output", CaptureConfig(sample_every_n_frames=2, stable_frames_required=2, blur_threshold=5, min_frames_between_saves=10))
    assert summary.frames_read == 20
    assert summary.candidates
    assert summary.candidates[0].path.is_file()
