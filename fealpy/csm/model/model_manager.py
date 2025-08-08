
from ...model import ModelManager

__all__ = ["CSMModelManager"]


class CSMModelManager(ModelManager):
    _registry = {
        "linear_elasticity": "fealpy.csm.model.linear_elasticity",
        "beam": "fealpy.csm.model.beam",
        "timoshenko_beam": "fealpy.csm.model.beam",
        "elastoplasticity": "fealpy.csm.model.elastoplasticity",
    }
