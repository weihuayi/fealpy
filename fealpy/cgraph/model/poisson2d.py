
from ..nodetype import CNodeType, PortConf, DataType


class Poisson2d(CNodeType):
    TITLE: str = "Poisson 2D"
    PATH: str = "model.poisson"
    INPUT_SLOTS = [
        PortConf("example", DataType.MENU, 0, items=[i for i in range(1, 9)])
    ]
    OUTPUT_SLOTS = [
        PortConf("domain", DataType.NONE),
        PortConf("solution", DataType.FUNCTION),
        PortConf("source", DataType.FUNCTION),
        PortConf("dirichlet", DataType.FUNCTION)
    ]

    @staticmethod
    def run(example):
        from fealpy.model import PDEModelManager
        mgr = PDEModelManager("poisson")
        model = mgr.get_example(int(example))
        return (model.domain(), ) + tuple(
            getattr(model, name)
            for name in ["solution", "source", "dirichlet"]
        )
