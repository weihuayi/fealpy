
from .nodetype import CNodeType, PortConf, DataType

__all__ = ["CGSolver"]


class CGSolver(CNodeType):
    TITLE: str = "CG 解法器"
    PATH: str = "解法器.迭代"
    INPUT_SLOTS = [
        PortConf("A", DataType.LINOPS, title="算子"),
        PortConf("b", DataType.TENSOR, title="向量"),
        PortConf("x0", DataType.TENSOR, title="初值", default=None),
        PortConf("maxit", DataType.INT, title="最大迭代数", default=10000, min_val=1)
    ]
    OUTPUT_SLOTS = [
        PortConf("out", DataType.TENSOR, title="结果")
    ]

    @staticmethod
    def run(*args, **kwargs):
        from ..solver import cg
        return cg(*args, **kwargs)


from .nodetype import CNodeType, PortConf, DataType

__all__ = ["DirectSolver"]

class DirectSolver(CNodeType):
    TITLE: str = "直接解法器"
    PATH: str = "解法器.直接"

    INPUT_SLOTS = [
        PortConf("A", DataType.LINOPS, title="矩阵"),
        PortConf("b", DataType.TENSOR, title="向量"),
        PortConf("matrix_type", DataType.MENU, title="矩阵类型",
                 default="G", items=["U", "L", "S", "SP", "G"]),
        PortConf("solver", DataType.MENU, title="求解器",
                 default="scipy", items=["scipy", "mumps", "pardiso", "cupy"])
    ]
    OUTPUT_SLOTS = [
        PortConf("out", DataType.TENSOR, title="结果")
    ]

    @staticmethod
    def run(*args, **kwargs):

        A = kwargs.get("A")
        b = kwargs.get("b")
        matrix_type = kwargs.get("matrix_type", kwargs.get("matrixType"))
        solver = kwargs.get("solver")
        from ..solver import DirectSolverManager
        mgr = DirectSolverManager()
        mgr.set_solver(solver_name=solver)
        mgr.set_matrix(A, matrix_type=matrix_type)
        x = mgr.solve(b)

        return x
    
from .nodetype import CNodeType, PortConf, DataType
__all__ = ["IterativeSolver"]
class IterativeSolver(CNodeType):
    TITLE: str = "迭代解法器"
    PATH: str = "解法器.迭代"
    INPUT_SLOTS = [
        PortConf("A", DataType.LINOPS, title="算子"),
        PortConf("b", DataType.TENSOR, title="向量"),
        PortConf("maxit", DataType.INT, title="最大迭代数", default=10000),
        PortConf("rtol", DataType.FLOAT, title="相对容忍度", default=1e-4),
        PortConf("atol", DataType.FLOAT, title="绝对容忍度", default=1e-4),
        PortConf("solver", DataType.MENU, title="求解器",
                 default="gmres", items=["gmres", "minres", "cg", "lgmres",  "bicg"])
    ]
    OUTPUT_SLOTS = [
        PortConf("out", DataType.TENSOR, title="结果")
    ]
    @staticmethod
    def run(*args, **kwargs):
        A = kwargs.get("A")
        b = kwargs.get("b")

        maxit = kwargs.get("maxit")
        rtol = kwargs.get("rtol")
        atol = kwargs.get("atol")
        solver = kwargs.get("solver")
        from ..solver import IterativeSolverManager
        mgr = IterativeSolverManager()
        mgr.set_solver(solver_name=solver)
        mgr.set_matrix(A)
        mgr.set_tolerances(rtol=rtol, atol=atol, maxit=maxit)
        x = mgr.solve(b)
        return x