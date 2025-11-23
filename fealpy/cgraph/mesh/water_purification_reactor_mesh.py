
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["WaterPurificationReactorMesher"]


class WaterPurificationReactorMesher(CNodeType):
    TITLE: str = "二维水净化反应器网格"
    PATH: str = "网格.构造"
    INPUT_SLOTS = [
        PortConf("block_length", DataType.FLOAT, 1, title="底面长方形长度", default=6.0),
        PortConf("block_width", DataType.FLOAT, 1, title="底面长方形宽度", default=2.0),
        PortConf("inlet_length", DataType.FLOAT, 1, title="入口长度", default=0.5),
        PortConf("inlet_width", DataType.FLOAT, 1, title="入口宽度", default=0.8),
        PortConf("gap", DataType.FLOAT, 1, title="窄缝长度", default=0.1),
        PortConf("gap_len", DataType.FLOAT, 1, title="窄缝宽度", default=1.0),
        PortConf("lc", DataType.FLOAT, 1, title="网格尺寸", default=0.4)
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import bm
        from fealpy.mesher import WPRMesher
        mesher = WPRMesher(options)
        mesher.generate()
        mesh = mesher.mesh

        return mesh
