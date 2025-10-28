
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["DirichletBC", "StructuralDirichletBC"]


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
    
class StructuralDirichletBC(DirichletBC):
    TITLE: str = "结构力学 Dirichlet 边界条件处理"
    PATH: str = "有限元.边界条件.结构力学"
    INPUT_SLOTS =[
        PortConf("gd", DataType.FUNCTION, 1, desc="Dirichlet 边界条件", title="边界条件"),
        PortConf("isDDof", DataType.TENSOR,1, desc="Dirichlet 自由度标记", title="Dirichlet 自由度"),
        PortConf("K", DataType.TENSOR,  desc="刚度矩阵", title="全局刚度矩阵"),
        PortConf("F", DataType.TENSOR,  desc="载荷向量", title="全局载荷向量"),
        PortConf("space", DataType.SPACE,  desc="函数空间", title="函数空间")
    ]
    OUTPUT_SLOTS = [
        PortConf("K", DataType.TENSOR,  desc="刚度矩阵", title="边界条件处理过的全局刚度矩阵"),
        PortConf("F", DataType.TENSOR,  desc="载荷向量", title="边界条件处理过的全局载荷向量"),
    ]
    
    @staticmethod
    def run(gd, isDDof, K, F, space):
        from fealpy.functionspace import TensorFunctionSpace
        tspace = TensorFunctionSpace(space, shape=(-1, 2))
        gdof = tspace.number_of_global_dofs()
        from fealpy.backend import bm
        threshold = bm.zeros(gdof, dtype=bool)
        threshold[isDDof()] = True
        from ...fem import DirichletBC
        bc = DirichletBC(space=tspace, gd=gd, threshold=threshold)
        K, F = bc.apply(K, F)

        return K, F