from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REQUIRED_MODULES = {
    "braceexpand": "braceexpand",
    "cv2": "opencv-python",
    "einops": "einops",
    "omegaconf": "omegaconf",
    "pyrender": "pyrender",
    "pytorch_lightning": "pytorch-lightning",
    "roma": "roma",
    "timm": "timm",
    "transformers": "transformers",
    "trimesh": "trimesh",
    "yacs": "yacs",
}


def get_requirements_path() -> Path:
    return Path(__file__).resolve().parent.parent / "requirements.txt"


def get_install_command() -> str:
    requirements = get_requirements_path()
    return f'"{sys.executable}" -m pip install -r "{requirements}"'


def get_missing_runtime_packages() -> list[str]:
    return sorted(
        package
        for module_name, package in REQUIRED_MODULES.items()
        if importlib.util.find_spec(module_name) is None
    )


def ensure_runtime_dependencies(feature: str = "this node") -> None:
    missing = get_missing_runtime_packages()
    if not missing:
        return

    joined = ", ".join(missing)
    raise RuntimeError(
        f"[sam3d-camshottoolkit] Missing runtime packages for {feature}: {joined}\n\n"
        f"Install them with:\n{get_install_command()}"
    )
