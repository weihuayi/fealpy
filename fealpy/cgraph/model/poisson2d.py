
from ..nodetype import CNodeType, PortConf, DataType


class Poisson2d(CNodeType):
    TITLE: str = "Poisson 2D"
    PATH: str = "model.poisson"
    INPUT_SLOTS = []
    OUTPUT_SLOTS = [
        PortConf("domain", DataType.NONE),
        PortConf("solution", DataType.FUNCTION),
        PortConf("source", DataType.FUNCTION),
        PortConf("dirichlet", DataType.FUNCTION)
    ]

    @staticmethod
    def run():
        from ...model.poisson.exp0002 import Exp0002
        model = Exp0002()
        return (model.domain(), ) + tuple(
            getattr(model, name)
            for name in ["solution", "source", "dirichlet"]
        )
