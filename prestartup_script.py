from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent
RUNTIME_DEPS_PATH = REPO_DIR / "nodes" / "runtime_deps.py"

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
