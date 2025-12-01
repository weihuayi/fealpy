
from ..nodetype import CNodeType, PortConf, DataType


class PoissonEquationDBC(CNodeType):
    
    TITLE: str = "Poisson 方程 (第一类边界条件)"
    PATH: str = "simulation.discretization"
    INPUT_SLOTS = [
        PortConf("space", DataType.SPACE, title="函数空间"),
        PortConf("q", DataType.INT, title="积分公式", default=3, min_val=1, max_val=17),
        PortConf("diffusion", DataType.FUNCTION, title="扩散系数", default=None),
        PortConf("source", DataType.FUNCTION, title="源", default=None),
        PortConf("gd", DataType.FUNCTION, title="边界条件")
    ]
    OUTPUT_SLOTS = [
        PortConf("operator", DataType.TENSOR, title="算子"),
        PortConf("source", DataType.TENSOR, title="源"),
    ]

    @staticmethod
    def run(space, q, diffusion, source, gd):
        from fealpy.fem import (
            LinearForm,
            BilinearForm,
            ScalarDiffusionIntegrator,
            ScalarSourceIntegrator
        )

        bform = BilinearForm(space)
        DI = ScalarDiffusionIntegrator(diffusion, q=q)
        bform.add_integrator(DI)
        A = bform.assembly()
        lform = LinearForm(space)
        SI = ScalarSourceIntegrator(source, q=q)
        lform.add_integrator(SI)
        F = lform.assembly()

        from fealpy.fem import DirichletBC
        BC = DirichletBC(space, gd=gd)
        A,F = BC.apply(A, F)

        return A, F
