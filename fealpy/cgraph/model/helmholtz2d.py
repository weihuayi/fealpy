
from ..nodetype import CNodeType, PortConf, DataType


class Helmholtz2d(CNodeType):
    TITLE: str = "二维 Helmholtz 问题模型"
    PATH: str = "模型.Helmholtz"
    INPUT_SLOTS = []
    OUTPUT_SLOTS = [
        PortConf("domain", DataType.NONE),
        PortConf("solution", DataType.FUNCTION),
        PortConf("source", DataType.FUNCTION),
        PortConf("dirichlet", DataType.FUNCTION)
    ]

    @staticmethod
    def run():
        from ...model.helmholtz.exp0004 import Exp0004
        model = Exp0004()
        return (model.domain(), ) + tuple(
            getattr(model, name)
            for name in ["solution", "source", "dirichlet"]
        )

