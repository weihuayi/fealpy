
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["StokesEquation"]


class StokesEquation(CNodeType):
    TITLE: str = "Stokes 方程 (第一类边界条件)"
    PATH: str = "有限元.方程离散"
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
    