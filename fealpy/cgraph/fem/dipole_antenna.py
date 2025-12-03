
from ..nodetype import CNodeType, PortConf, DataType
from ...fem import DirichletBCOperator
from fealpy.functionspace import FirstNedelecFESpace


__all__ = ["DipoleAntennaEquation"]

class DipoleAntennaEquation(CNodeType):
    TITLE: str = "麦克斯韦方程离散处理"
    PATH: str = "simulation.discretization"
    DESC: str = "麦克斯韦方程的离散形式"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, title="网格"),
        PortConf("q", DataType.INT, title ="积分公式", default=3, min_val=1, max_val=17),
        PortConf("diffusion", DataType.FUNCTION, title="扩散系数"),
        PortConf("reaction", DataType.FUNCTION, title="反应系数"),
        PortConf("source", DataType.FUNCTION, title="源项"),
        PortConf("Y", DataType.FUNCTION, title="阻抗边界条件的系数"),
        PortConf("ID", DataType.TENSOR, title="阻抗边界自由度"),
        PortConf("gd", DataType.FUNCTION, title="Dirichlet边界条件"),
        PortConf("isDDof", DataType.TENSOR, title="Dirichlet边界自由度")

    ]
    OUTPUT_SLOTS = [
        PortConf("A", DataType.LINOPS),
        PortConf("F", DataType.TENSOR),
    ]

    @staticmethod
    def run(mesh, q: int, diffusion, reaction, source, Y, ID, gd, isDDof):
        from ...fem import (
            BilinearForm,
            LinearForm,
            ScalarMassIntegrator,
            CurlCurlIntegrator,
            BoundaryFaceMassIntegrator,
            ScalarSourceIntegrator
        )
        space = FirstNedelecFESpace(mesh, p=1)
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
        F = lform.assembly()

        dbc = DirichletBCOperator(form=bform, gd=gd, isDDof=isDDof)
        uh = dbc.init_solution()
        F = dbc.apply(F, uh)
        return dbc, F
