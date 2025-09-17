
from ..nodetype import CNodeType, PortConf, DataType


class PoissonEquationDBC(CNodeType):
    TITLE: str = "Poisson 方程 (第一类边界条件)"
    PATH: str = "有限元.方程离散"
    INPUT_SLOTS = [
        PortConf("space", DataType.SPACE, title="函数空间"),
        PortConf("q", DataType.INT, title="积分公式", default=3, min_val=1, max_val=17),
        PortConf("diffusion", DataType.FUNCTION, title="扩散系数", default=None),
        PortConf("source", DataType.FUNCTION, title="源", default=None),
        PortConf("gd", DataType.FUNCTION, title="边界条件")
    ]
    OUTPUT_SLOTS = [
        PortConf("operator", DataType.LINOPS, title="算子"),
        PortConf("source", DataType.TENSOR, title="源"),
        PortConf("uh", DataType.TENSOR, title="初始解")
    ]

    @staticmethod
    def run(space, q, diffusion, source, gd):
        from ...fem import (
            LinearForm,
            BilinearForm,
            ScalarDiffusionIntegrator,
            ScalarSourceIntegrator
        )

        bform = BilinearForm(space)
        DI = ScalarDiffusionIntegrator(diffusion, q=q)
        bform.add_integrator(DI)

        lform = LinearForm(space)
        SI = ScalarSourceIntegrator(source, q=q)
        lform.add_integrator(SI)
        F = lform.assembly()

        from ...fem import DirichletBCOperator

        isDDof = space.is_boundary_dof()
        dbc = DirichletBCOperator(form=bform, gd=gd, isDDof=isDDof)
        uh = dbc.init_solution()
        F = dbc.apply(F, uh)

        return dbc, F, uh
