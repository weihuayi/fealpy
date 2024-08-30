from .elastic_material import ElasticMaterial
import numpy as np

class HyperElasticMaterial(ElasticMaterial):
    def __init__(self, name, model='Neo-Hookean'):
        super().__init__(name)
        self.set_property('shear_modulus', None)
        self.set_property('bulk_modulus', None)
        self.model = model

    def stress(self, strain):
        """计算超弹性材料的应力，根据选定的模型"""
        if self.model == 'Neo-Hookean':
            return self.neo_hookean_stress(strain)
        elif self.model == 'Mooney-Rivlin':
            return self.mooney_rivlin_stress(strain)
        else:
            raise ValueError(f"Unknown hyperelastic model: {self.model}")

    def neo_hookean_stress(self, strain):
        """Neo-Hookean 模型应力计算"""
        mu = self.get_property('shear_modulus')
        J = np.linalg.det(strain)  # 体积变化比
        b = strain @ strain.T  # 左Cauchy-Green应变张量
        stress = mu * (b - np.eye(3)) / J
        return stress

    def mooney_rivlin_stress(self, strain):
        """Mooney-Rivlin 模型应力计算（可根据具体需求实现）"""
        raise NotImplementedError("Mooney-Rivlin stress calculation needs implementation.")

