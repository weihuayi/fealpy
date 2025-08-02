# 1. Python built-in imports
import os
import sys
import importlib.util

# 2. Third-party imports
import pkg_resources


def get_fcos_resource(relative_path: str) -> str:
    """
    Retrieves the absolute path to a resource file within the FCOS resources directory.
    
    This function constructs the path relative to the FEALPy package's vision/fcos_resources
    directory and returns the absolute path using pkg_resources.
    
    Parameters:
        relative_path (str): Relative path to the resource file within the FCOS resources.
        
    Returns:
        str: Absolute filesystem path to the requested resource.
    """
    resource_path = os.path.join('vision', 'fcos_resources', relative_path)
    return pkg_resources.resource_filename('fealpy', resource_path)


def get_fcos_customization(relative_path: str) -> str:
    """
    Retrieves the absolute path to a customization file within the FCOS customizations directory.
    
    This function constructs the path relative to the FEALPy package's vision/fcos_customizations
    directory and returns the absolute path using pkg_resources.
    
    Parameters:
        relative_path (str): Relative path to the customization file within the FCOS customizations.
        
    Returns:
        str: Absolute filesystem path to the requested customization file.
    """
    customization_path = os.path.join('vision', 'fcos_customizations', relative_path)
    return pkg_resources.resource_filename('fealpy', customization_path)


def load_fcos_customizations() -> None:
    """
    Dynamically loads custom modifications to FCOS core modules.
    
    This function overrides the following FCOS modules with custom implementations:
    1. fcos_core.config.path_catalog
    2. fcos_core.data.datasets.coco
    
    The custom implementations are loaded from the FEALPy package's vision/fcos_customizations
    directory. If the custom files are not found, the original FCOS implementations will be used.
    """
    # Load custom path_catalog implementation
    path_catalog_path = get_fcos_customization('path_catalog.py')
    if os.path.exists(path_catalog_path):
        spec = importlib.util.spec_from_file_location(
            "fcos_core.config.path_catalog", 
            path_catalog_path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["fcos_core.config.path_catalog"] = module
        spec.loader.exec_module(module)
    
    # Load custom coco dataset implementation
    coco_path = get_fcos_customization('coco.py')
    if os.path.exists(coco_path):
        spec = importlib.util.spec_from_file_location(
            "fcos_core.data.datasets.coco", 
            coco_path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["fcos_core.data.datasets.coco"] = module
        spec.loader.exec_module(module)


# Execute customizations when module is imported
load_fcos_customizations()