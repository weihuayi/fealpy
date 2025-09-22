
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["DirichletBC"]


class DirichletBC(CNodeType):
    TITLE: str = "Dirichlet 边界条件处理"
    PATH: str = "有限元.边界条件"
    INPUT_SLOTS = [
        PortConf("gd", DataType.FUNCTION),
        PortConf("isDDof", DataType.TENSOR),
        PortConf("form", DataType.LINOPS),
        PortConf("F", DataType.TENSOR)
    ]
    OUTPUT_SLOTS = [
        PortConf("A", DataType.LINOPS),
        PortConf("F", DataType.TENSOR),
        PortConf("uh", DataType.TENSOR)
    ]

    @staticmethod
    def run(gd, isDDof, form, F):
        from ...fem import DirichletBCOperator
        dbc = DirichletBCOperator(form=form, gd=gd, isDDof=isDDof)
        uh = dbc.init_solution()
        F = dbc.apply(F, uh)
        return dbc, F, uh
