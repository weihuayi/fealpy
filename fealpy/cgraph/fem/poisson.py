
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["PoissonEquation"]


class PoissonEquation(CNodeType):
    TITLE: str = "Poisson Equation"
    PATH: str = "fem.presets"
    INPUT_SLOTS = [
        PortConf("space", DataType.SPACE),
        PortConf("q", DataType.INT, default=3, min_val=1, max_val=17),
        PortConf("diffusion", DataType.FUNCTION),
        PortConf("source", DataType.FUNCTION)
    ]
    OUTPUT_SLOTS = [
        PortConf("operator", DataType.LINOPS),
        PortConf("source", DataType.TENSOR)
    ]

    @staticmethod
    def run(space, q: int, diffusion, source):
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

        return bform, lform.assembly()
