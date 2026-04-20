"""Minimal SAM3D camera-shot toolkit for ComfyUI."""

from comfy_env import wrap_nodes
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

wrap_nodes()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
