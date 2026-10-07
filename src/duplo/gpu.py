"""Detectie en installatie van CUDA-ondersteuning voor torch."""
import subprocess
import sys

CUDA_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"


def cuda_available() -> bool:
    probe = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
        capture_output=True, text=True,
    )
    return probe.returncode == 0 and probe.stdout.strip() == "True"


def ensure_cuda_torch() -> None:
    """Installeert de CUDA-build van torch als die nog niet aanwezig is.

    Draait in een subprocess zodat een reeds geladen CPU-build het vervangen
    van de DLL's niet blokkeert; importeer torch/ultralytics pas hierna.
    """
    if cuda_available():
        return
    print("CUDA-torch niet gevonden; CUDA-build van torch wordt geinstalleerd "
          "(kan enkele minuten duren)...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "-U", "torch", "torchvision",
        "--index-url", CUDA_TORCH_INDEX,
    ])
    if result.returncode != 0 or not cuda_available():
        raise SystemExit(
            "GPU gevraagd, maar CUDA is niet beschikbaar "
            "(installatie mislukt of geen NVIDIA-GPU aanwezig)."
        )
