
from .nodetype import CNodeType, PortConf, DataType

__all__ = ["CGSolver", "DLDSolver"]


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


class DLDSolver(CNodeType):
    TITLE: str = "DLD Solver"
    PATH: str = "solver.direct"
    INPUT_SLOTS = [
        PortConf("A", DataType.LINOPS),
        PortConf("b", DataType.TENSOR),
        PortConf("uspace", DataType.SPACE)
    ]
    OUTPUT_SLOTS = [
        PortConf("uh", DataType.TENSOR),
        PortConf("ph", DataType.TENSOR)
    ]

    @staticmethod
    def run(A, b, uspace):
        from fealpy.solver import spsolve
        x = spsolve(A, b)
        ugdof = uspace.number_of_global_dofs()
        uh = x[:ugdof]
        ph = x[ugdof:]

        uspace.mesh.nodedata['ph'] = ph
        uspace.mesh.nodedata['uh'] = uh.reshape(2,-1).T
        uspace.mesh.to_vtk('dld_chip910.vtu')
        return uh ,ph