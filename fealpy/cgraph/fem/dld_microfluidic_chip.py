
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["DLDMicroflidicChipEquation"]


class DLDMicroflidicChipEquation(CNodeType):
    TITLE: str = "DLD Microflidic Chip Equation"
    PATH: str = "fem.presets"
    INPUT_SLOTS = [
        PortConf("uspace", DataType.SPACE),
        PortConf("pspace", DataType.SPACE),
        PortConf("velocity_dirichlet", DataType.FUNCTION),
        PortConf("pressure_dirichlet", DataType.FUNCTION),
        PortConf("is_velocity_boundary", DataType.FUNCTION),
        PortConf("is_pressure_boundary", DataType.FUNCTION)
    ]
    OUTPUT_SLOTS = [
        PortConf("bform", DataType.TENSOR),
        PortConf("lform", DataType.TENSOR)
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
        apply_bc = BC.apply
        A, F = apply_bc(A, F)


        return A, F
    