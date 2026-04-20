# SAM3D Body ComfyUI Cam Shot Toolkit

Minimal ComfyUI custom node package for:

- `Load SAM 3D Body Model`
- `SAM 3D Body: Process Image`
- `SAM 3D Body: Render Offset View`

This repo is a focused extraction from a larger SAM3DBody node pack, trimmed for calibrated pose-to-camera-shot workflows.

## Install

Place this repo in `ComfyUI/custom_nodes/` and restart ComfyUI.

This minimal package currently runs in ComfyUI's active Python environment rather than a separate `comfy-env` isolated runtime.

## Included

- SAM3D model loader with Hugging Face auto-download
- single-image SAM3D processing node
- calibrated offset renderer with pivot, background, and lighting controls
