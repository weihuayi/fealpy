from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["LinearElasticityEigenEquation"]

class LinearElasticityEigenEquation(CNodeType):
    TITLE: str = "线弹性特征值方程（矩阵装配）"
    PATH: str = "有限元.方程离散"
    INPUT_SLOTS = [
        PortConf("space", DataType.SPACE, "函数空间"),
        PortConf("q", DataType.INT, title="积分精度", default=3, min_val=1, max_val=17),
        PortConf("material", DataType.MENU, title="材料属性"),
        PortConf("displacement_bc", DataType.FUNCTION, title="边界条件函数"),
        PortConf("is_displacement_boundary", DataType.FUNCTION, title="边界标识函数"),
    ]
    OUTPUT_SLOTS = [
        PortConf("stiffness", DataType.LINOPS, title="刚度矩阵S"),
        PortConf("mass", DataType.LINOPS, title="质量矩阵M"),
    ]

    @staticmethod
    def run(space, q: int, material, displacement_bc, is_displacement_boundary):

        from ...fem import BilinearForm
        from ...fem import LinearElasticityIntegrator
        from ...fem import ScalarMassIntegrator as MassIntegrator
        from ...fem import DirichletBC
        from ...backend import backend_manager as bm

        
        bform_S = BilinearForm(space)
        elst = LinearElasticityIntegrator(material, q=q)
        elst.assembly.set('fast')
        bform_S.add_integrator(elst)
        S = bform_S.assembly()

        bform_M = BilinearForm(space)
        mass = MassIntegrator(material.density, q=q)
        bform_M.add_integrator(mass)
        M = bform_M.assembly()
 
        bc = DirichletBC(
                space,
                gd=displacement_bc,
                threshold=is_displacement_boundary)
        isFreeDof = bm.logical_not(bc.is_boundary_dof)
        S = S.to_scipy()[isFreeDof, :][:, isFreeDof]
        M = M.to_scipy()[isFreeDof, :][:, isFreeDof]

        return S, M
