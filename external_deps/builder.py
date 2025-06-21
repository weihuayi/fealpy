import os
import sys
import platform
import json
import importlib


# add the current directory to the path 
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def build_ext():
    """Build the extension modules for the current platform.
    """
    ext_modules = []
    # Load json file
    with open(os.path.join(os.path.dirname(__file__), "config.json")) as f:
        deps = json.load(f)

    for dep in deps:
        if os.getenv(dep["env"]): # Check if the environment variable is set
            module = importlib.import_module(dep["name"])
            module.build(dep, ext_modules)
    return ext_modules
