
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["DipoleAntennaEquation"]

class DipoleAntennaEquation(CNodeType):
    TITLE: str = "麦克斯韦 方程"
    PATH: str = "有限元.方程离散"
    DESC: str = "麦克斯韦方程的离散形式"
    INPUT_SLOTS = [
        PortConf("space", DataType.SPACE, desc="函数空间"),
        PortConf("q", DataType.INT, desc="积分公式", default=3, min_val=1, max_val=17),
        PortConf("diffusion", DataType.FUNCTION, desc="扩散系数"),
        PortConf("reaction", DataType.FUNCTION, desc="反应系数"),
        PortConf("source", DataType.FUNCTION, desc="源项"),
        PortConf("Y", DataType.FUNCTION, desc="阻抗边界条件的系数"),
        PortConf("ID", DataType.TENSOR, desc="阻抗边界自由度")

    ]
    OUTPUT_SLOTS = [
        PortConf("operator", DataType.LINOPS, desc="算子"),
        PortConf("source", DataType.TENSOR, desc="源")
    ]

    @staticmethod
    def run(space, q: int, diffusion, reaction, source, Y, ID):
        from ...fem import (
            BilinearForm,
            LinearForm,
            ScalarMassIntegrator,
            CurlCurlIntegrator,
            BoundaryFaceMassIntegrator,
            ScalarSourceIntegrator
        )

        bform = BilinearForm(space)
        DI = CurlCurlIntegrator(diffusion, q=q)
        DM = ScalarMassIntegrator(reaction, q=q)
        DN = BoundaryFaceMassIntegrator(threshold=ID, coef=Y, q=q)

        bform.add_integrator(DI)
        bform.add_integrator(DM)
        bform.add_integrator(DN)

        lform = LinearForm(space)
        SI = ScalarSourceIntegrator(source, q=q)
        lform.add_integrator(SI)

        return bform, lform.assembly()