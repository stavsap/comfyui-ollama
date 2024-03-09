import importlib.util
import sys
import os
import pathlib

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

name = "test.py"
spec = importlib.util.spec_from_file_location(name, os.path.join(pathlib.Path(__file__).parent.resolve(),name))
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
