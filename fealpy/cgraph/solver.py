
from .nodetype import CNodeType, PortConf, DataType

__all__ = ["CGSolver"]


class CGSolver(CNodeType):
    TITLE: str = "CG Solver"
    PATH: str = "solver.iterative"
    INPUT_SLOTS = [
        PortConf("A", DataType.LINOPS),
        PortConf("b", DataType.TENSOR),
        PortConf("x0", DataType.TENSOR, default=None),
        PortConf("maxit", DataType.INT, default=10000, min_val=1)
    ]
    OUTPUT_SLOTS = [
        PortConf("out", DataType.TENSOR)
    ]

    @staticmethod
    def run(*args, **kwargs):
        from ..solver import cg
        return cg(*args, **kwargs)
