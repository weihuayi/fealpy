
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["LinearElasticityEquation"]


class LinearElasticityEquation(CNodeType):
    TITLE: str = "LinearElasticity Equation"
    PATH: str = "fem.presets"
    INPUT_SLOTS = [
        PortConf("space", DataType.SPACE),
        PortConf("q", DataType.INT, default=3, min_val=1, max_val=17),
        PortConf("body_force", DataType.FUNCTION),
        PortConf("lam", DataType.FLOAT),
        PortConf("mu", DataType.FLOAT),
        PortConf("hypo", DataType.STRING, default='plane_strain', items=['plane_strain', 'plane_stress', '3D'])
    ]
    OUTPUT_SLOTS = [
        PortConf("operator", DataType.LINOPS),
        PortConf("body_force", DataType.TENSOR)
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
