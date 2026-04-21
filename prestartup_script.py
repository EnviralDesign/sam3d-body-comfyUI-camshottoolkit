from __future__ import annotations

import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from nodes.runtime_deps import get_install_command, get_missing_runtime_packages


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
