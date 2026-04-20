from __future__ import annotations

import importlib.util


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


def main() -> None:
    missing = [
        package
        for module_name, package in REQUIRED_MODULES.items()
        if importlib.util.find_spec(module_name) is None
    ]

    if missing:
        joined = ", ".join(sorted(missing))
        print(
            "[sam3d-camshottoolkit] Missing runtime packages detected: "
            f"{joined}. Run install.py or `python -m pip install -r requirements.txt`."
        )
    else:
        print("[sam3d-camshottoolkit] Prestartup complete.")


main()
