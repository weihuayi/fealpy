
from ..nodetype import CNodeType, PortConf, DataType


class DLDMicroflidicChip2D(CNodeType):
    TITLE: str = "DLD Microflidic Chip 2D"
    PATH: str = "model.dld_microfluidic_chip"
    INPUT_SLOTS = []
    OUTPUT_SLOTS = [
        PortConf("velocity_dirichlet", DataType.FUNCTION),
        PortConf("pressure_dirichlet", DataType.FUNCTION),
        PortConf("is_velocity_boundary", DataType.FUNCTION),
        PortConf("is_pressure_boundary", DataType.FUNCTION)
    ]

    @staticmethod
    def run():
        from fealpy.cfd.model.stationary_incompressible_navier_stokes.exp0003 import Exp0003
        model = Exp0003()
        print(1)
        return tuple(
            getattr(model, name)
            for name in ["velocity_dirichlet", "pressure_dirichlet", "is_velocity_boundary", "is_pressure_boundary"]
        )

