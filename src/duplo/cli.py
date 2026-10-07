"""Command-line interface voor de pipeline-scripts."""
import argparse

from .config import DEVICES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--device", choices=DEVICES,
                        help="cpu of gpu; overschrijft 'device' uit de config")
    return parser.parse_args()
