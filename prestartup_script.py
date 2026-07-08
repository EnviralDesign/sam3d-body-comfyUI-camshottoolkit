from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent
RUNTIME_DEPS_PATH = REPO_DIR / "nodes" / "runtime_deps.py"


def configure_transformers_flash_attn_mapping() -> None:
    """
    Work around Transformers builds that know about flash-attn checks but do not
    include the import-utils distribution mapping for flash_attn.

    Keep this prestartup hook cheap: do not import Transformers just to patch the
    mapping. The detector load path applies the same patch before importing
    SAM3-specific classes.
    """
    import_utils = sys.modules.get("transformers.utils.import_utils")
    if import_utils is None:
        return

    package_mapping = getattr(import_utils, "PACKAGE_DISTRIBUTION_MAPPING", None)
    if package_mapping is None:
        return

    if "flash_attn" in package_mapping:
        return

    package_mapping["flash_attn"] = ["flash-attn"]
    print("[sam3d-camshottoolkit] Patched Transformers flash_attn distribution mapping.")


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
