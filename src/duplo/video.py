"""Video-inlezen met frame-stride en voortgangsbalk."""
import time
from collections.abc import Iterator

import cv2
import numpy as np
from tqdm import tqdm


def video_fps(video_path: str) -> float:
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    video.release()
    return fps


class VideoFrames:
    """Itereert videoframes volgens een vaste stride, met voortgangsweergave.

    In een terminal verschijnt een balk die op één regel bijwerkt; wordt de
    uitvoer vastgelegd (logbestand, IDE-paneel), dan verschijnen regels bij
    elke 10% of elke 5 seconden, direct geflusht.

    Yields (frame_idx, frame) voor elk frame waarop frame_idx % stride == 0;
    na de iteratie bevat frames_read het aantal ingelezen frames.
    """

    def __init__(self, video_path: str, stride: int = 1, desc: str = "Video") -> None:
        self.video_path = video_path
        self.stride = stride
        self.desc = desc
        self.frames_read = 0

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        video = cv2.VideoCapture(self.video_path)
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = tqdm(total=total if total > 0 else None, desc=self.desc,
                        unit="frame", disable=None)
        report_step = max(1, total // 10) if total > 0 else 0
        last_report = time.monotonic()
        try:
            frame_idx = 0
            while True:
                ok, frame = video.read()
                if not ok:
                    break
                self.frames_read = frame_idx + 1
                progress.update(1)
                if progress.disable and report_step:
                    now = time.monotonic()
                    if (self.frames_read % report_step == 0
                            or now - last_report >= 5.0):
                        last_report = now
                        percent = self.frames_read * 100 // total
                        print(f"{self.desc}: {percent}% "
                              f"({self.frames_read}/{total} frames)", flush=True)
                if frame_idx % self.stride == 0:
                    yield frame_idx, frame
                frame_idx += 1
        finally:
            progress.close()
            video.release()
