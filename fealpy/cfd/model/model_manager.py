from ...model import ModelManager

__all__ = ['CFDPDEModelManager', 'CFDTestModelManager']

class CFDPDEModelManager(ModelManager):
    _registry = {
        "stationary_incompressible_navier_stokes": "fealpy.cfd.model.stationary_incompressible_navier_stokes"
    }

class CFDTestModelManager(ModelManager):
    _registry = {
        "stationary_incompressible_navier_stokes": "fealpy.cfd.model.test.stationary_incompressible_navier_stokes",
        "incompressible_navier_stokes": "fealpy.cfd.model.test.incompressible_navier_stokes",
        "stationary_stokes": "fealpy.cfd.model.test.stationary_stokes"
    }
