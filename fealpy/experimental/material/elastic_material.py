from ..backend import backend_manager as bm

from builtins import float, str
from .material_base import MaterialBase
from fealpy.experimental.fem.utils import shear_strain, normal_strain
from ..functionspace.utils import flatten_indices

from ..typing import TensorLike

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
    def __init__(self, name: str, 
            elastic_modulus: float = 1, poisson_ratio:float = 0.3, 
            hypo: str = "3D" ):
        super().__init__(name)
        self.hypo = hypo
        self.set_property('elastic_modulus', elastic_modulus)
        self.set_property('poisson_ratio', poisson_ratio)

        E = self.get_property('elastic_modulus')
        nu = self.get_property('poisson_ratio')
        lam = self.calculate_lame_lambda()
        mu = self.calculate_shear_modulus()

        if hypo == "3D":
            self.D = bm.tensor([[2 * mu + lam, lam, lam, 0, 0, 0],
                                [lam, 2 * mu + lam, lam, 0, 0, 0],
                                [lam, lam, 2 * mu + lam, 0, 0, 0],
                                [0, 0, 0, mu, 0, 0],
                                [0, 0, 0, 0, mu, 0],
                                [0, 0, 0, 0, 0, mu]], dtype=bm.float64)
        elif hypo == "plane_stress":
            self.D = E / (1 - nu ** 2) * bm.array([[1, nu, 0], 
                                                   [nu, 1, 0], 
                                                   [0, 0, (1 - nu) / 2]], dtype=bm.float64)
        elif hypo == "plane_strain":
            self.D = bm.tensor([[2 * mu + lam, lam, 0],
                                [lam, 2 * mu + lam, 0],
                                [0, 0, mu]], dtype=bm.float64)
        else:
            raise NotImplementedError("Only 3D, plane_stress, and plane_strain are supported.")

    def elastic_matrix(self) -> TensorLike:
        """
        Calculate the elastic matrix D based on the defined hypothesis (3D, plane stress, or plane strain).

        Returns:
            TensorLike: The elastic matrix D.
                - For 2D problems (GD=2): (1, 1, 3, 3)
                - For 3D problems (GD=3): (1, 1, 6, 6)
            Here, the first dimension (NC) is the number of cells, and the second dimension (NQ) is the 
            number of quadrature points, both of which are set to 1 for compatibility with other finite 
            element tensor operations.
        """
        return self.D[None, None, ...]
    
    def strain_matrix(self, dof_priority: bool, gphi: TensorLike) -> TensorLike:
        '''
        Constructs the strain-displacement matrix B for the material based on the gradient of the shape functions.

        Parameters:
            dof_priority (bool): A flag that determines the ordering of DOFs.
                                If True, the priority is given to the first dimension of degrees of freedom.
            gphi (TensorLike): A tensor representing the gradient of the shape functions. Its shape
                            typically includes the number of local degrees of freedom and the geometric 
                            dimension (GD).
        
        Returns:
            TensorLike: The strain-displacement matrix `B`, which is a tensor with shape:
                        - For 2D problems (GD=2): (NC, NQ, 3, tldof)
                        - For 3D problems (GD=3): (NC, NQ, 6, tldof)
                        Here, NC is the number of cells, NQ is the number of quadrature points, 
                        and tldof is the number of local degrees of freedom.
        '''
        ldof, GD = gphi.shape[-2:]
        if dof_priority:
            indices = flatten_indices((ldof, GD), (1, 0))
        else:
            indices = flatten_indices((ldof, GD), (0, 1))
        B = bm.concat([normal_strain(gphi, indices),
                       shear_strain(gphi, indices)], axis=-2)
        return B
