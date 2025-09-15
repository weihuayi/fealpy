
from ..nodetype import CNodeType, PortConf, DataType


class Poisson2d(CNodeType):
    TITLE: str = "二维 Poisson 问题模型"
    PATH: str = "model.poisson"
    INPUT_SLOTS = [
        PortConf("example", DataType.MENU, 0, title="例子编号", default=1, items=[i for i in range(1, 9)])
    ]
    OUTPUT_SLOTS = [
        PortConf("domain", DataType.NONE, title="区域"),
        PortConf("solution", DataType.FUNCTION, title="解函数"),
        PortConf("source", DataType.FUNCTION, title="源函数"),
        PortConf("dirichlet", DataType.FUNCTION, title="边界条件函数")
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
