
from ...model import ModelManager

__all__ = ["CSMModelManager"]


class CSMModelManager(ModelManager):
    _registry = {
        "linear_elasticity": "fealpy.csm.model.linear_elasticity",
        "beam": "fealpy.csm.model.beam",
        "timobeam_axle": "fealpy.csm.model.beam",
        "elastoplasticity": "fealpy.csm.model.elastoplasticity",
        "truss": "fealpy.csm.model.truss",
        "truss_tower": "fealpy.csm.model.truss",
        "channel_beam": "fealpy.csm.model.beam"
    }
