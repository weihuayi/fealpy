from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["StationaryNS2d"]

class StationaryNS2d(CNodeType):
    TITLE: str = "Stationary NS 2D"
    PATH: str = "cfd.model.test.stationary_navier_stokes"
    INPUT_SLOTS = []
    OUTPUT_SLOTS = [
        PortConf("mu", DataType.FLOAT),
        PortConf("rho", DataType.FLOAT),
        PortConf("domain", DataType.DOMAIN),
        PortConf("velocity", DataType.FUNCTION),
        PortConf("pressure", DataType.FUNCTION),
        PortConf("source", DataType.FUNCTION),
        PortConf("velocity_dirichlet", DataType.FUNCTION),
        PortConf("pressure_dirichlet", DataType.FUNCTION),
        PortConf("is_velocity_boundary", DataType.FUNCTION),
        PortConf("is_pressure_boundary", DataType.FUNCTION)
    ]

    @staticmethod
    def run() -> Union[object]:
        from fealpy.cfd.model.test.stationary_incompressible_navier_stokes.exp0001 import Exp0001

        model = Exp0001()
        return (model.mu, model.rho, model.domain()) + tuple(
            getattr(model, name)
            for name in ["velocity", "pressure", "source", "velocity_dirichlet", "pressure_dirichlet", "is_velocity_boundary", "is_pressure_boundary"]
        )