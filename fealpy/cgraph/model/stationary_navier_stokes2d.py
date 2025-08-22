from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["StationaryNS2d"]

class StationaryNS2d(CNodeType):
    TITLE: str = "Stationary NS 2D"
    PATH: str = "model.navier_stokes"
    INPUT_SLOTS = [
        PortConf("rho", DataType.FLOAT, default=1.0, min_val=0.0, desc="密度参数"),
        PortConf("mu", DataType.FLOAT, default=1.0, min_val=0.0, desc="粘性系数")
    ]
    OUTPUT_SLOTS = [
        PortConf("pde", DataType.NONE)
    ]

    @staticmethod
    def run(rho: float = 1.0, mu: float = 1.0) -> Union[object]:
        from fealpy.cfd.model.test.stationary_incompressible_navier_stokes.stationary_incompressible_navier_stokes_2d import FromSympy

        model = FromSympy(rho=rho, mu=mu)
        model.select_pde["poly2d"]()

        return model