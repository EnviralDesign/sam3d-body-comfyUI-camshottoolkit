# SAM3D Body ComfyUI Cam Shot Toolkit

Minimal ComfyUI custom node package for:

- `Cam Shot Toolkit: Load SAM3D Model`
- `Cam Shot Toolkit: Load SAM3 Person Detector`
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
- SAM3 person detector loader with official gated repo support and ungated
  mirror fallback for prompt-based multi-person detection
- single-image SAM3D processing node
- person selection for SAM3D outputs: `person_index=-1` uses all detected people,
  while `0..N` selects a specific detected person. When no mask is connected,
  nonzero selection modes automatically run a torchvision person detector so
  multi-person inputs produce multiple SAM3D crops.
- calibrated offset renderer with pivot, background, and lighting controls
- interactive browser viewer for scouting the render camera with orbit, pan, roll, and dolly controls
- optional render `auto` mode that ignores saved viewer camera state and realigns from each SAM3D input for API workflows

## Node Categories

- `CamShotToolkit`
- `CamShotToolkit/processing`
- `CamShotToolkit/visualization`

## Notes

- Model weights are downloaded automatically on first use into `ComfyUI/models/sam3dbody`.
- SAM3 detector weights are downloaded into `ComfyUI/models/sam3_person_detector`.
  The detector loader tries `facebook/sam3` first in `auto` mode and falls back
  to an ungated mirror such as `jetjodh/sam3` if official access is gated.
- The render node uses `pyrender` in Python, not a Three.js viewport.
- On macOS, install OSMesa first with `brew install osmesa`; the package defaults
  PyOpenGL to `osmesa` to avoid pyglet/Cocoa worker-thread crashes in ComfyUI.
- On headless Linux GPU hosts, the package defaults PyOpenGL to EGL during
  Comfy prestartup so offscreen rendering works without an X display.
- `install.py` is intentionally a no-op for Registry compliance. Dependency installation should be handled by ComfyUI-Manager or an explicit `pip install -r requirements.txt`.

## Donations & Support

If this saves you time, you can support the work here:

- [Patreon](https://www.patreon.com/EnviralDesign)
- [GitHub Sponsors](https://github.com/sponsors/EnviralDesign)
- [PayPal](https://www.paypal.com/donate?hosted_button_id=RP8EJAHSDTZ86)
