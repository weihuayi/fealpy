
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["StationaryNS2d"]

class StationaryNSBC(CNodeType):
    TITLE: str = "StationaryNSBC"
    PATH: str = "cfd.bc"
    INPUT_SLOTS = [
        PortConf("uspace", DataType.SPACE),
        PortConf("pspace", DataType.SPACE),
        PortConf("velocity_dirichlet", DataType.FUNCTION),
        PortConf("pressure_dirichlet", DataType.FUNCTION),
        PortConf("is_velocity_boundary", DataType.FUNCTION),
        PortConf("is_pressure_boundary", DataType.FUNCTION)
    ]
    OUTPUT_SLOTS = [
        PortConf("apply_bc", DataType.FUNCTION)
    ]
    @staticmethod
    def run(uspace, pspace, velocity_dirichlet, pressure_dirichlet, is_velocity_boundary, is_pressure_boundary):
        from fealpy.fem import DirichletBC
        BC = DirichletBC(
            (uspace, pspace), 
            gd=(velocity_dirichlet, pressure_dirichlet), 
            threshold=(is_velocity_boundary, is_pressure_boundary),
            method='interp')
        apply_bc = BC.apply
        return apply_bc