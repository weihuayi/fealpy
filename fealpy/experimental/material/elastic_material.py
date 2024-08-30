from .material_base import MaterialBase
import numpy as np

class ElasticMaterial(MaterialBase):
    def __init__(self, name):
        super().__init__(name)
        self.set_property('elastic_modulus', None)
        self.set_property('poisson_ratio', None)

    def calculate_shear_modulus(self):
        E = self.get_property('elastic_modulus')
        nu = self.get_property('poisson_ratio')
        if E is not None and nu is not None:
            return E / (2 * (1 + nu))
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

    def constitutive_matrix(self):
        """计算二维平面应力状态下的本构矩阵 D"""
        E = self.get_property('elastic_modulus')
        nu = self.get_property('poisson_ratio')
        if E is None or nu is None:
            raise ValueError("Elastic modulus and Poisson's ratio must be defined.")

        # 计算平面应力状态的本构矩阵
        D = E / (1 - nu ** 2) * np.array([[1, nu, 0],
                                          [nu, 1, 0],
                                          [0, 0, (1 - nu) / 2]])
        return D
