# SAM3D Body ComfyUI Cam Shot Toolkit

Minimal ComfyUI custom node package for:

- `Cam Shot Toolkit: Load SAM3D Model`
- `Cam Shot Toolkit: Process Image`
- `Cam Shot Toolkit: Render Offset View`

This repo is a focused extraction from a larger SAM3DBody node pack, trimmed for calibrated pose-to-camera-shot workflows.

## Install

1. Clone or copy this repo into `ComfyUI/custom_nodes/`.
2. Install dependencies with ComfyUI-Manager or manually:

```bash
python -m pip install -r requirements.txt
```

3. Restart ComfyUI.

This package installs into ComfyUI's active Python environment. It does not attempt to replace ComfyUI's existing `torch` / `torchvision` install. A working ComfyUI CUDA environment is expected already.

## Registry Publishing

This repository is prepared for Comfy Registry publishing.

Before publishing:

1. Set your real `PublisherId` in `pyproject.toml`.
2. Create a Registry publishing API key for that publisher.
3. Publish with:

```bash
comfy node publish
```

## Included

- SAM3D model loader with Hugging Face auto-download
- single-image SAM3D processing node
- calibrated offset renderer with pivot, background, and lighting controls

## Node Categories

- `CamShotToolkit`
- `CamShotToolkit/processing`
- `CamShotToolkit/visualization`

## Notes

- Model weights are downloaded automatically on first use into `ComfyUI/models/sam3dbody`.
- The render node uses `pyrender` in Python, not a Three.js viewport.
- `install.py` is intentionally a no-op for Registry compliance. Dependency installation should be handled by ComfyUI-Manager or an explicit `pip install -r requirements.txt`.
