
from ..nodetype import CNodeType, PortConf, DataType


class PoissonEquationDBC(CNodeType):
    TITLE: str = "Poisson Equation Dirichlet BC"
    PATH: str = "fem.presets"
    INPUT_SLOTS = [
        PortConf("space", DataType.SPACE),
        PortConf("q", DataType.INT, default=3, min_val=1, max_val=17),
        PortConf("diffusion", DataType.FUNCTION),
        PortConf("source", DataType.FUNCTION),
        PortConf("gd", DataType.FUNCTION)
    ]
    OUTPUT_SLOTS = [
        PortConf("operator", DataType.LINOPS),
        PortConf("source", DataType.TENSOR),
        PortConf("uh", DataType.TENSOR)
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
