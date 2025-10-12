
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["LinearElasticityEquation"]


class LinearElasticityEquation(CNodeType):
    TITLE: str = "线弹性方程"
    PATH: str = "有限元.方程离散"
    INPUT_SLOTS = [
        PortConf("space", DataType.SPACE, "函数空间"),
        PortConf("q", DataType.INT, title="积分公式", default=3, min_val=1, max_val=17),
        PortConf("body_force", DataType.FUNCTION, title="载荷函数"),
        PortConf("lam", DataType.FLOAT, title="拉梅常数"),
        PortConf("mu", DataType.FLOAT, title="剪切模量"),
        PortConf("hypo", DataType.MENU, 0, title="力学假设", default='plane_strain', items=['plane_strain', 'plane_stress', '3D'])
    ]
    OUTPUT_SLOTS = [
        PortConf("operator", DataType.LINOPS, title="算子"),
        PortConf("body_force", DataType.TENSOR, title="载荷")
    ]

    @staticmethod
    def run(space, q: int, lam, mu, hypo, body_force):
        from ...fem import (
            LinearForm,
            BilinearForm,
            LinearElasticityIntegrator,
            ScalarSourceIntegrator,
        )
        from ...material import LinearElasticMaterial

        LEM = LinearElasticMaterial(
            name='E1nu025',
            lame_lambda=lam, shear_modulus=mu,
            hypo=hypo)

        bform = BilinearForm(space)
        DI = LinearElasticityIntegrator(LEM, q=q)
        bform.add_integrator(DI)

        lform = LinearForm(space)
        SI = ScalarSourceIntegrator(body_force, q=q)
        lform.add_integrator(SI)

        return bform, lform.assembly()
