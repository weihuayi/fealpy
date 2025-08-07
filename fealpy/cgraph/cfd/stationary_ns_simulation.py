
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["StationaryNS2d"]


class StationaryNSSimulation(CNodeType):
    TITLE: str = "StationaryNSSimulation"
    PATH: str = "cfd.simulation"
    INPUT_SLOTS = [
        PortConf("equation", DataType.NONE)
    ]
    OUTPUT_SLOTS = [
        PortConf("simulation", DataType.NONE),
    ]
    @staticmethod
    def run(equation):
        from fealpy.cfd.simulation.fem.stationary_incompressible_ns import Newton
        simulation = Newton(equation)
        return simulation