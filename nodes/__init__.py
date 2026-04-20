"""Minimal node registry for the cam shot toolkit."""

from .processing.load_model import NODE_CLASS_MAPPINGS as LOAD_MAPPINGS
from .processing.load_model import NODE_DISPLAY_NAME_MAPPINGS as LOAD_DISPLAY_MAPPINGS
from .processing.process import NODE_CLASS_MAPPINGS as PROCESS_MAPPINGS
from .processing.process import NODE_DISPLAY_NAME_MAPPINGS as PROCESS_DISPLAY_MAPPINGS
from .processing.visualize import NODE_CLASS_MAPPINGS as VIS_MAPPINGS
from .processing.visualize import NODE_DISPLAY_NAME_MAPPINGS as VIS_DISPLAY_MAPPINGS

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for mappings in [LOAD_MAPPINGS, PROCESS_MAPPINGS, VIS_MAPPINGS]:
    NODE_CLASS_MAPPINGS.update(mappings)

for mappings in [LOAD_DISPLAY_MAPPINGS, PROCESS_DISPLAY_MAPPINGS, VIS_DISPLAY_MAPPINGS]:
    NODE_DISPLAY_NAME_MAPPINGS.update(mappings)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
