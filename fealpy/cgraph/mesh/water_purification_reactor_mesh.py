
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["WaterPurificationReactorMesher"]


class WaterPurificationReactorMesher(CNodeType):
    TITLE: str = "水净化反应器网格"
    PATH: str = "preprocess.mesher"
    INPUT_SLOTS = [
        PortConf("block_length", DataType.FLOAT, 1, title="底面长方形长度", default=6.0),
        PortConf("block_width", DataType.FLOAT, 1, title="底面长方形宽度", default=2.0),
        PortConf("inlet_length", DataType.FLOAT, 1, title="入口长度", default=0.5),
        PortConf("inlet_width", DataType.FLOAT, 1, title="入口宽度", default=0.8),
        PortConf("gap", DataType.FLOAT, 1, title="窄缝长度", default=0.1),
        PortConf("gap_len", DataType.FLOAT, 1, title="窄缝宽度", default=1.0),
        PortConf("lc", DataType.FLOAT, 1, title="底面三角形网格尺寸", default=0.4),
        PortConf("interval", DataType.NONE, 1, title="区间网格区域", default=[0, 0.4]),
        PortConf("nx", DataType.FLOAT, 1, title="区间网格分段数", default=8)
    ]
    OUTPUT_SLOTS = [
        PortConf("tmesh", DataType.MESH, title="三角形网格"),
        PortConf("imesh", DataType.MESH, title="区间网格")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import bm
        from fealpy.mesher import WPRMesher
        from fealpy.mesh import IntervalMesh
        mesher = WPRMesher(options)
        mesher.generate()
        tmesh = mesher.mesh

        interval = options['interval']
        nx = options['nx']
        imesh = IntervalMesh.from_interval_domain(interval, nx)

        return tmesh, imesh
