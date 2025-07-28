from ...model import ModelManager


__all__ = ['CFDTestModelManager']


class CFDTestModelManager(ModelManager):
    _registry = {
        "stationary_incompressible_navier_stokes": "fealpy.cfd.model.test.stationary_incompressible_navier_stokes",
        "incompressible_navier_stokes": "fealpy.cfd.model.test.incompressible_navier_stokes"
    }
