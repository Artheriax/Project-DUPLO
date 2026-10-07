"""Vaste schrijfconventies voor resultaten (CSV en voorbeeldframes)."""
from pathlib import Path

import cv2
import pandas as pd


def prepare_output_dir(cfg: dict) -> Path:
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_rows_csv(out_dir: Path, filename: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(out_dir / filename, index=False)


def save_screenshot(out_dir: Path, prefix: str, frame_idx: int, image) -> None:
    cv2.imwrite(str(out_dir / f"{prefix}_{frame_idx:05d}.jpg"), image)
