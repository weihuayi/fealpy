
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["StokesEquation"]


class StokesEquation(CNodeType):
    r"""Stokes equations with Dirichlet boundary conditions.

    Inputs:
        uspace(space): Function space for the velocity field.
        pspace(space): Function space for the pressure field.
        velocity_dirichlet (function): Dirichlet boundary condition for velocity.
        pressure_dirichlet (function): Dirichlet boundary condition for pressure.
        is_velocity_boundary (function): Predicate function identifying velocity boundary regions.
        is_pressure_boundary (function): Predicate function identifying pressure boundary regions.
    
    Outputs:
        bform (tensor): Assembled system operator.
        lform (tensor): Assembled right-hand side vector.
    """
    TITLE: str = "Stokes 方程 (第一类边界条件)"
    PATH: str = "simulation.discretization"
    DESC: str = """该节点构建并组装带Dirichlet边界条件的Stokes方程有限元离散系统, 
                输出包含系统算子与右端项向量，用于稳态粘性流动求解。"""
    INPUT_SLOTS = [
        PortConf("uspace", DataType.SPACE, title="速度函数空间"),
        PortConf("pspace", DataType.SPACE, title="压力函数空间"),
        PortConf("velocity_dirichlet", DataType.FUNCTION, title="速度边界条件"),
        PortConf("pressure_dirichlet", DataType.FUNCTION, title="压力边界条件"),
        PortConf("is_velocity_boundary", DataType.FUNCTION, title="速度边界"),
        PortConf("is_pressure_boundary", DataType.FUNCTION, title="压力边界")
    ]
    OUTPUT_SLOTS = [
        PortConf("bform", DataType.TENSOR, title="算子"),
        PortConf("lform", DataType.TENSOR, title="向量")
    ]

    @staticmethod
    def run(uspace, pspace, velocity_dirichlet, pressure_dirichlet, is_velocity_boundary, is_pressure_boundary):
        from fealpy.fem import LinearForm, BilinearForm, BlockForm, LinearBlockForm
        from fealpy.fem import ScalarDiffusionIntegrator as DiffusionIntegrator
        from fealpy.fem import PressWorkIntegrator

        A00 = BilinearForm(uspace)
        BD = DiffusionIntegrator()
        BD.coef = 1.0
        A00.add_integrator(BD)
        A01 = BilinearForm((pspace, uspace))
        BP = PressWorkIntegrator()
        BP.coef = -1.0
        A01.add_integrator(BP)
        bform = BlockForm([[A00, A01], [A01.T, None]])

        L0 = LinearForm(uspace)
        L1 = LinearForm(pspace)
        lform = LinearBlockForm([L0, L1])

        from fealpy.fem import DirichletBC
        A = bform.assembly()
        F = lform.assembly()
        BC = DirichletBC(
            (uspace, pspace), 
            gd=(velocity_dirichlet, pressure_dirichlet), 
            threshold=(is_velocity_boundary, is_pressure_boundary),
            method='interp')
        A, F = BC.apply(A, F)


        return A, F
    