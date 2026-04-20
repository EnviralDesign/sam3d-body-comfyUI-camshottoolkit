from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    requirements = repo_dir / "requirements.txt"

    if not requirements.exists():
        print("[sam3d-camshottoolkit] No requirements.txt found; skipping install.")
        return

    cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements)]
    print(f"[sam3d-camshottoolkit] Installing runtime dependencies from {requirements}...")
    subprocess.check_call(cmd, cwd=repo_dir)
    print("[sam3d-camshottoolkit] Dependency install complete.")


if __name__ == "__main__":
    main()
