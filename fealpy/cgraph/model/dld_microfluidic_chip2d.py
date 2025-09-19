
from ..nodetype import CNodeType, PortConf, DataType


class DLDMicrofluidicChip2D(CNodeType):
    TITLE: str = "DLD Microfluidic Chip 2D"
    PATH: str = "model.dld_microfluidic_chip"
    INPUT_SLOTS = [
        PortConf("radius", DataType.FLOAT),
        PortConf("centers", DataType.FLOAT),
        PortConf("inlet_boundary", DataType.TENSOR),
        PortConf("outlet_boundary", DataType.TENSOR),
        PortConf("wall_boundary", DataType.TENSOR)
    ]
    OUTPUT_SLOTS = [
        PortConf("velocity_dirichlet", DataType.FUNCTION),
        PortConf("pressure_dirichlet", DataType.FUNCTION),
        PortConf("is_velocity_boundary", DataType.FUNCTION),
        PortConf("is_pressure_boundary", DataType.FUNCTION)
    ]

    @staticmethod
    def run(**options):
        from fealpy.model.dld_microfluidic_chip.exp0001 import Exp0001
        model = Exp0001(**options)
        return tuple(
            getattr(model, name)
            for name in ["velocity_dirichlet", "pressure_dirichlet", "is_velocity_boundary", "is_pressure_boundary"]
        )

