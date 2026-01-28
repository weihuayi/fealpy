from ..nodetype import CNodeType, PortConf, DataType

class DipoleAntenna3D(CNodeType):
    TITLE: str = "三维 偶极子天线 问题模型"
    PATH: str = "模型.curlcurl"
    DESC: str = "三维 偶极子天线 问题模型"
    INPUT_SLOTS = [
        PortConf("r0", DataType.FLOAT, title="开始时球体半径", default=1.9),
        PortConf("r1", DataType.FLOAT, title="结束时球体半径", default=2.4),
        PortConf("mu", DataType.FLOAT, title="磁导率", default=1),
        PortConf("epsilon", DataType.FLOAT, title="介电常数", default=1),
        PortConf("s", DataType.FLOAT, title="多项式系数", default=5.0),
        PortConf("p", DataType.FLOAT, title="多项式次数", default=2)
    ]
    OUTPUT_SLOTS = [
        PortConf("Y", DataType.FLOAT, title="阻抗边界系数"),
        PortConf("diffusion", DataType.FUNCTION, title="扩散边界系数"),
        PortConf("reaction", DataType.FUNCTION, title="反应边界系数"),
        PortConf("source", DataType.FUNCTION, title="源项"),
        PortConf("dirichlet", DataType.FUNCTION, title="Dirichlet边界条件")
    ]

    @staticmethod
    def run(**options):
        from fealpy.model.curlcurl.exp0003 import Exp0003
        model = Exp0003(**options)
        return (model.Y, ) + tuple(
            getattr(model, name)
            for name in ["diffusion", "reaction", "source", "dirichlet"])