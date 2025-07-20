
from ...model import ModelManager

__all__ = ["CSMModelManager"]


class CSMModelManager(ModelManager):
    _registry = {
        "beam": "fealpy.csm.model.beam",
        "elastoplasticity": "fealpy.csm.model.elastoplasticity",
    }
