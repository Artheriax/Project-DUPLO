"""Video-encoding voor debug-video's in een breed afspeelbaar formaat.

OpenCV's ingebouwde mp4v-codec (MPEG-4 Part 2) speelt niet af in Chromium
(de VS Code-videospeler en browsers) en wordt door deelplatforms geweigerd.
Deze writer encodeert daarom H.264 (yuv420p) met faststart, via de
ffmpeg-binary die met imageio-ffmpeg wordt meegeleverd.
"""
import subprocess
from pathlib import Path

import cv2
import numpy as np


class DebugVideoWriter:
    """Schrijft een H.264-mp4 die in VS Code, browsers en deelplatforms werkt.

    Valt terug op OpenCV's mp4v als de ffmpeg-binary ontbreekt, zodat de
    pipeline altijd een debug-video oplevert. Zonder size worden de
    afmetingen bij het eerste frame bepaald, zodat de pipelines de writer
    niet vooraf hoeven te openen.
    """

    def __init__(self, path: Path, fps: float,
                 size: tuple[int, int] | None = None) -> None:
        self._path = path
        self._fps = fps
        self._process = None
        self._writer = None
        if size is not None:
            self._open(size)

    def _open(self, size: tuple[int, int]) -> None:
        ffmpeg = self._find_ffmpeg()
        if ffmpeg is not None:
            width, height = size
            self._process = subprocess.Popen(
                [ffmpeg, "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{width}x{height}", "-r", str(self._fps), "-i", "-",
                 "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 str(self._path)],
                stdin=subprocess.PIPE,
            )
        else:
            print("Waarschuwing: imageio-ffmpeg ontbreekt; de debug-video "
                  "krijgt het mp4v-formaat dat niet overal afspeelt.")
            self._writer = cv2.VideoWriter(
                str(self._path), cv2.VideoWriter_fourcc(*"mp4v"), self._fps,
                size)

    @staticmethod
    def _find_ffmpeg() -> str | None:
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            return None

    def write(self, frame: np.ndarray) -> None:
        if self._process is None and self._writer is None:
            height, width = frame.shape[:2]
            self._open((width, height))
        if self._process is not None:
            self._process.stdin.write(frame.tobytes())
        else:
            self._writer.write(frame)

    def release(self) -> None:
        if self._process is not None:
            self._process.stdin.close()
            self._process.wait()
            if self._process.returncode != 0:
                print(f"Waarschuwing: ffmpeg stopte met code "
                      f"{self._process.returncode}; de debug-video is "
                      f"mogelijk incompleet.")
            self._process = None
        elif self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "DebugVideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
