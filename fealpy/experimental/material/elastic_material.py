from ..backend import backend_manager as bm

from .material_base import MaterialBase

class ElasticMaterial(MaterialBase):
    def __init__(self, name):
        super().__init__(name)
        self.set_property('elastic_modulus', None)
        self.set_property('poisson_ratio', None)

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
    def __init__(self, name):
        super().__init__(name)

    def plane_stress_elastic_matrix(self):
        """
        Calculate the elastic matrix D for plane stress condition.
        """
        E = self.get_property('elastic_modulus')
        nu = self.get_property('poisson_ratio')
        if E is None or nu is None:
            raise ValueError("Elastic modulus and Poisson's ratio must be defined.")

        D = E / (1 - nu ** 2) * bm.array([[1, nu, 0],
                                          [nu, 1, 0],
                                          [0, 0, (1 - nu) / 2]], dtype=bm.float64)
        return D
    
    def plane_strain_elastic_matrix(self):
        """
        Calculate the elastic matrix D for plane strain condition.
        """
        lam = self.calculate_lame_lambda()
        mu = self.calculate_shear_modulus
        if lam is None or mu is None:
            raise ValueError("Lame and Shear modulus must be defined.")

        D = bm.tensor([[2 * mu + lam, lam, 0],
                    [lam, 2 * mu + lam, 0],
                    [0, 0, mu]], dtype=bm.float64)
        return D
    
    def elastic_matrix(self):
        """
        Calculate the elastic matrix D for three-dimensional elasticity.
        """
        lam = self.calculate_lame_lambda()
        mu = self.calculate_shear_modulus
        if lam is None or mu is None:
            raise ValueError("Lame and Shear modulus must be defined.")

        D = bm.tensor([[2 * mu + lam, lam, lam, 0, 0, 0],
                                  [lam, 2 * mu + lam, lam, 0, 0, 0],
                                  [lam, lam, 2 * mu + lam, 0, 0, 0],
                                  [0, 0, 0, mu, 0, 0],
                                  [0, 0, 0, 0, mu, 0],
                                  [0, 0, 0, 0, 0, mu]], dtype=bm.float64)
        return D
