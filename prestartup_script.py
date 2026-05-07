from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent
RUNTIME_DEPS_PATH = REPO_DIR / "nodes" / "runtime_deps.py"


def configure_headless_opengl() -> None:
    """
    Prefer an offscreen PyOpenGL backend for pyrender when ComfyUI starts.

    PyOpenGL chooses its platform the first time OpenGL/pyrender is imported, so
    this has to run during Comfy prestartup. Respect explicit user choices.
    """
    if os.environ.get("PYOPENGL_PLATFORM"):
        return
    if sys.platform == "darwin":
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        print("[sam3d-camshottoolkit] macOS detected; using PYOPENGL_PLATFORM=osmesa.")
        return
    if os.name == "nt":
        return
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    print("[sam3d-camshottoolkit] Headless display detected; using PYOPENGL_PLATFORM=egl.")


def configure_transformers_flash_attn_mapping() -> None:
    """
    Work around Transformers builds that know about flash-attn checks but do not
    include the import-utils distribution mapping for flash_attn.
    """
    try:
        from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
    except Exception:
        return

    if "flash_attn" in PACKAGE_DISTRIBUTION_MAPPING:
        return

    PACKAGE_DISTRIBUTION_MAPPING["flash_attn"] = ["flash-attn"]
    print("[sam3d-camshottoolkit] Patched Transformers flash_attn distribution mapping.")


configure_headless_opengl()
configure_transformers_flash_attn_mapping()

spec = importlib.util.spec_from_file_location("sam3d_camshottoolkit_runtime_deps", RUNTIME_DEPS_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"[sam3d-camshottoolkit] Could not load runtime deps helper from {RUNTIME_DEPS_PATH}")

runtime_deps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime_deps)
get_install_command = runtime_deps.get_install_command
get_missing_runtime_packages = runtime_deps.get_missing_runtime_packages


def main() -> None:
    missing = get_missing_runtime_packages()

    if missing:
        joined = ", ".join(sorted(missing))
        print(
            "[sam3d-camshottoolkit] Missing runtime packages detected: "
            f"{joined}.\n[sam3d-camshottoolkit] Install with:\n{get_install_command()}"
        )
    else:
        print("[sam3d-camshottoolkit] Prestartup complete.")


main()
