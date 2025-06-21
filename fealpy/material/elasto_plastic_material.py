from .elastic_material import ElasticMaterial
import numpy as np

class ElastoPlasticMaterial(ElasticMaterial):
    def __init__(self, name):
        super().__init__(name)
        self.set_property('yield_stress', None)
        self.set_property('hardening_modulus', None)

    def constitutive_relation(self, strain):
        """计算弹塑性材料的应力-应变关系"""
        stress, plastic_strain = self.elastic_plastic_response(strain)
        return stress

    def elastic_plastic_response(self, strain):
        """实现弹塑性应力-应变计算，假设线性硬化规则"""
        E = self.get_property('elastic_modulus')
        sigma_y = self.get_property('yield_stress')
        H = self.get_property('hardening_modulus')

        # 计算试验应力
        trial_stress = E * strain
        if np.abs(trial_stress) <= sigma_y:
            # 弹性区域
            return trial_stress, 0
        else:
            # 塑性区域，考虑硬化效应
            plastic_strain = (np.abs(trial_stress) - sigma_y) / (E + H)
            stress = np.sign(trial_stress) * (sigma_y + H * plastic_strain)
            return stress, plastic_strain

