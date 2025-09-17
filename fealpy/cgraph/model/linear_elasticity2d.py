
from ..nodetype import CNodeType, PortConf, DataType


class LinearElasticity2d(CNodeType):
    TITLE: str = "二维线弹性问题模型"
    PATH: str = "模型.线弹性"
    INPUT_SLOTS = []
    OUTPUT_SLOTS = [
        PortConf("domain", DataType.NONE),
        PortConf("lam", DataType.FLOAT),
        PortConf("mu", DataType.FLOAT),
        PortConf("displacement", DataType.FUNCTION),
        PortConf("body_force", DataType.FUNCTION),
        PortConf("displacement_bc", DataType.FUNCTION),
        PortConf("hypo", DataType.STRING)
    ]

    @staticmethod
    def run():
        from ...model.linear_elasticity.exp0002 import Exp0002
        model = Exp0002()
        return (model.domain(), model.lam(), model.mu()) + tuple(
            getattr(model, name)
            for name in ["displacement", "body_force", "displacement_bc", "hypo"]
        )
