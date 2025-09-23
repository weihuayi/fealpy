
from ..nodetype import CNodeType, PortConf, DataType


class DLDMicrofluidicChip2D(CNodeType):
    TITLE: str = "二维 DLD 微流控芯片流体数学模型"
    PATH: str = "模型.DLD 微流控芯片"
    INPUT_SLOTS = [
        PortConf("radius", DataType.FLOAT, title="微柱半径"),
        PortConf("centers", DataType.TENSOR, title="微柱中心坐标数组"),
        PortConf("inlet_boundary", DataType.TENSOR, title="入口边界"),
        PortConf("outlet_boundary", DataType.TENSOR, title="出口边界"),
        PortConf("wall_boundary", DataType.TENSOR, title="通道壁面边界")
    ]
    OUTPUT_SLOTS = [
        PortConf("velocity_dirichlet", DataType.FUNCTION, title="速度边界条件"),
        PortConf("pressure_dirichlet", DataType.FUNCTION, title="压力边界条件"),
        PortConf("is_velocity_boundary", DataType.FUNCTION, title="速度边界"),
        PortConf("is_pressure_boundary", DataType.FUNCTION, title="压力边界")
    ]

    @staticmethod
    def run(**options):
        from fealpy.model.dld_microfluidic_chip.exp0001 import Exp0001
        model = Exp0001(**options)
        return tuple(
            getattr(model, name)
            for name in ["velocity_dirichlet", "pressure_dirichlet", "is_velocity_boundary", "is_pressure_boundary"]
        )

