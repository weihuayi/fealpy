from ..backend import backend_manager as bm

from builtins import float, str
from .material_base import MaterialBase
from fealpy.experimental.fem.utils import shear_strain, normal_strain
from ..functionspace.utils import flatten_indices

from ..typing import TensorLike
from typing import Optional

class ElasticMaterial(MaterialBase):
    def __init__(self, name):
        super().__init__(name)

    def calculate_elastic_modulus(self):
        lam = self.get_property('lame_lambda')
        mu = self.get_property('shear_modulus')
        if lam is not None and mu is not None:
            E = mu * (3 * lam + 2 * mu) / (lam + mu)
            return E
        else:
            raise ValueError("Lame's lambda and shear modulus must be defined.")
        
    def calculate_poisson_ratio(self):
        lam = self.get_property('lame_lambda')
        mu = self.get_property('shear_modulus')
        if lam is not None and mu is not None:
            nu = lam / (2 * (lam + mu))
            return nu
        else:
            raise ValueError("Lame's lambda and shear modulus must be defined.")

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
            elastic_modulus: Optional[float] = None, 
            poisson_ratio: Optional[float] = None, 
            lame_lambda: Optional[float] = None, 
            shear_modulus: Optional[float] = None,
            hypo: str = "3D" ):
        super().__init__(name)
        self.hypo = hypo

        # self.set_property('elastic_modulus', elastic_modulus)
        # self.set_property('poisson_ratio', poisson_ratio)

        if elastic_modulus is not None and poisson_ratio is not None and lame_lambda is None and shear_modulus is None:
            self.set_property('elastic_modulus', elastic_modulus)
            self.set_property('poisson_ratio', poisson_ratio)
            self.set_property('lame_lambda', self.calculate_lame_lambda())
            self.set_property('shear_modulus', self.calculate_shear_modulus())

        elif lame_lambda is not None and shear_modulus is not None and elastic_modulus is None and poisson_ratio is None:
            self.set_property('lame_lambda', lame_lambda)
            self.set_property('shear_modulus', shear_modulus)
            self.set_property('elastic_modulus', self.calculate_elastic_modulus())
            self.set_property('poisson_ratio', self.calculate_poisson_ratio())

        elif lame_lambda is not None and shear_modulus is not None and elastic_modulus is not None and poisson_ratio is not None:
            calculated_E = self.calculate_elastic_modulus()
            calculated_nu = self.calculate_poisson_ratio()
            if abs(calculated_E - elastic_modulus) > 1e-5 or abs(calculated_nu - poisson_ratio) > 1e-5:
                raise ValueError("The input elastic modulus and Poisson's ratio are inconsistent with "
                                 "the values calculated from the provided Lame's lambda and shear modulus.")
            self.set_property('elastic_modulus', elastic_modulus)
            self.set_property('poisson_ratio', poisson_ratio)
            self.set_property('lame_lambda', lame_lambda)
            self.set_property('shear_modulus', shear_modulus)

        else:
            raise ValueError("You must provide either (elastic_modulus, poisson_ratio) or (lame_lambda, shear_modulus), or all four.")


        # if lame_lambda is not None:
        #     self.set_property('lame_lambda', lame_lambda)
        # if shear_modulus is not None:
        #     self.set_property('shear_modulus', shear_modulus)

        # if lame_lambda is not None and shear_modulus is not None:
        #     calculated_E = self.calculate_elastic_modulus()
        #     calculated_nu = self.calculate_poisson_ratio()
        #     if abs(calculated_E - elastic_modulus) > 1e-5 or abs(calculated_nu - poisson_ratio) > 1e-5:
        #         raise ValueError("The input elastic modulus and Poisson's ratio are inconsistent with "
        #                          "the values calculated from the provided Lame's lambda and shear modulus.")
        #     self.set_property('elastic_modulus', calculated_E)
        #     self.set_property('poisson_ratio', calculated_nu)
        
        # if lame_lambda is None or shear_modulus is None:
        #     calculated_lam = self.calculate_lame_lambda()
        #     calculated_mu = self.calculate_shear_modulus()
        #     self.set_property('lame_lambda', calculated_lam)
        #     self.set_property('shear_modulus', calculated_mu)

        self.E = self.get_property('elastic_modulus')
        self.nu = self.get_property('poisson_ratio')
        self.lam = self.get_property('lame_lambda')
        self.mu = self.get_property('shear_modulus')

        E = self.E
        nu = self.nu
        lam = self.lam
        mu = self.mu

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

    def elastic_matrix(self, bcs: Optional[TensorLike] = None) -> TensorLike:
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

    def strain_value(self):
        pass

    def stress_value(self):
        pass