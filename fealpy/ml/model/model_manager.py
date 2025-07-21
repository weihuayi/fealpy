import importlib
import os
import sys
from ...model.model_manager import ModelManager


class PDEModelManager(ModelManager):
    """PDEModelManager manages PDE model types and examples.

    Examples:
        >>> PDEModelManager.show_types()
        >>> manager = PDEModelManager('poisson')
        >>> manager.show_examples()
        >>> model = manager.get_example(1)
    """
    _registry = {
        "poisson": "fealpy.ml.model.poisson",
        "diffusion_reaction": "fealpy.ml.model.diffusion_reaction",

    }