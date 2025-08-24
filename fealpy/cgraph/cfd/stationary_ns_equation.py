
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["StationaryNS2d"]

class StationaryNSEquation(CNodeType):
    TITLE: str = "StationaryNSEquation"
    PATH: str = "cfd.equation"
    INPUT_SLOTS = [
        PortConf("pde", DataType.NONE)
    ]
    OUTPUT_SLOTS = [
        PortConf("equation", DataType.NONE)
    ]

    @staticmethod
    def run(pde):
        from fealpy.cfd.equation.stationary_incompressible_ns import StationaryIncompressibleNS
        equation = StationaryIncompressibleNS(pde)  
        return equation


