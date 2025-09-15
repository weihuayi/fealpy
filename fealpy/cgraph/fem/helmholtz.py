
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["HelmholtzEquation"]


class HelmholtzEquation(CNodeType):
    TITLE: str = "Helmholtz 方程"
    PATH: str = "fem.presets"
    INPUT_SLOTS = [
        PortConf("space", DataType.SPACE, title="函数空间"),
        PortConf("q", DataType.INT, title="积分公式", default=3, min_val=1, max_val=17),
        PortConf("diffusion", DataType.FUNCTION, title="扩散系数", default=1.),
        PortConf("reaction", DataType.FLOAT, title="反应系数", default=-1.),
        PortConf("source", DataType.FUNCTION, title="源")
    ]
    OUTPUT_SLOTS = [
        PortConf("operator", DataType.LINOPS, title="算子"),
        PortConf("source", DataType.TENSOR, title="源")
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
