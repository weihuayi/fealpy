from ..backend import backend_manager as bm

from .material_base import MaterialBase

class ElasticMaterial(MaterialBase):
    def __init__(self, name):
        super().__init__(name)

    def calculate_shear_modulus(self):
        E = self.get_property('elastic_modulus')
        nu = self.get_property('poisson_ratio')
        if E is not None and nu is not None:
            mu = E / (2 * (1 + nu))
            return mu
        else:
            raise ValueError("Elastic modulus and Poisson's ratio must be defined.")
        
    def calculate_lame_lambda(self):
        E = self.get_property('elastic_modulus')
        nu = self.get_property('poisson_ratio')
        if E is not None and nu is not None:
            lam = nu * E / ((1 + nu) * (1 - 2 * nu))
            return lam
        else:
            raise ValueError("Elastic modulus and Poisson's ratio must be defined.")

    def calculate_bulk_modulus(self):
        E = self.get_property('elastic_modulus')
        nu = self.get_property('poisson_ratio')
        if E is not None and nu is not None:
            return E / (3 * (1 - 2 * nu))
        else:
            raise ValueError("Elastic modulus and Poisson's ratio must be defined.")

class LinearElasticMaterial(ElasticMaterial):
    def __init__(self, name, 
            elastic_modulus: float=1, poisson_ratio:float=0.3, 
            hypo: str = "3D" ):
        super().__init__(name)
        self.set_property('elastic_modulus', elastic_modulus)
        self.set_property('poisson_ratio', poisson_ratio)
        self.hypothesis = hypothesis

        lam = self.calculate_lame_lambda()
        mu = self.calculate_shear_modulus

        if hypo == "3D":
            self.D = bm.tensor([[2 * mu + lam, lam, lam, 0, 0, 0],
                                      [lam, 2 * mu + lam, lam, 0, 0, 0],
                                      [lam, lam, 2 * mu + lam, 0, 0, 0],
                                      [0, 0, 0, mu, 0, 0],
                                      [0, 0, 0, 0, mu, 0],
                                      [0, 0, 0, 0, 0, mu]], dtype=bm.float64)
        elif hypo == "plane_stress":
            E = self.get_property('elastic_modulus')
            nu = self.get_property('poisson_ratio')
            self.D = E / (1 - nu ** 2) * bm.array([
                [1, nu, 0], 
                [nu, 1, 0], 
                [0, 0, (1 - nu) / 2]], dtype=bm.float64)
        elif hypo == "plane_strain":
            self.D = bm.tensor([[2 * mu + lam, lam, 0],
                        [lam, 2 * mu + lam, 0],
                        [0, 0, mu]], dtype=bm.float64)
        else:
            raise ""

    def elastic_matrix(self):
        """
        Calculate the elastic matrix D for three-dimensional elasticity.
        """
        return self.D[None, None, ...]
