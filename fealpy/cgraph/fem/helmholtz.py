
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["HelmholtzEquation"]


class HelmholtzEquation(CNodeType):
    TITLE: str = "Helmholtz Equation"
    PATH: str = "fem.presets"
    INPUT_SLOTS = [
        PortConf("space", DataType.SPACE),
        PortConf("q", DataType.INT, default=3, min_val=1, max_val=17),
        PortConf("diffusion", DataType.FUNCTION, default=1.),
        PortConf("reaction", DataType.FLOAT, default=-1.),
        PortConf("source", DataType.FUNCTION)
    ]
    OUTPUT_SLOTS = [
        PortConf("operator", DataType.LINOPS),
        PortConf("source", DataType.TENSOR)
    ]

    @staticmethod
    def run(space, q: int, diffusion, reaction, source):
        from ...fem import (
            LinearForm,
            BilinearForm,
            ScalarDiffusionIntegrator,
            ScalarMassIntegrator,
            ScalarSourceIntegrator
        )

        bform = BilinearForm(space)
        DI = ScalarDiffusionIntegrator(diffusion, q=q)
        DM = ScalarMassIntegrator(reaction, q=q)
        bform.add_integrator(DI)
        bform.add_integrator(DM)

        lform = LinearForm(space)
        SI = ScalarSourceIntegrator(source, q=q)
        lform.add_integrator(SI)

        return bform, lform.assembly()
    