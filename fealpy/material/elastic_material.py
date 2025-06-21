from ..backend import backend_manager as bm

from builtins import float, str
from .material_base import MaterialBase
from ..functionspace.utils import flatten_indices

from ..typing import TensorLike
from typing import Optional, Tuple, List

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
            hypo: str = "3D",
            device: str = None):
        
        super().__init__(name)
        self.hypo = hypo
        self.device = device

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
                                [0, 0, 0, 0, 0, mu]], dtype=bm.float64, device=device)
        elif hypo == "plane_stress":
            self.D = E / (1 - nu ** 2) * bm.array([[1, nu, 0], 
                                                   [nu, 1, 0], 
                                                   [0, 0, (1 - nu) / 2]], dtype=bm.float64, device=device)
        elif hypo == "plane_strain":
            self.D = bm.tensor([[2 * mu + lam, lam, 0],
                                [lam, 2 * mu + lam, 0],
                                [0, 0, mu]], dtype=bm.float64, device=device)
        else:
            raise NotImplementedError("Only 3D, plane_stress, and plane_strain are supported.")
    
    @property
    def elastic_modulus(self) -> float:
        """获取弹性模量"""
        return self.E

    @property
    def poisson_ratio(self) -> float:
        """获取泊松比"""
        return self.nu

    @property
    def lame_lambda(self) -> float:
        """获取拉梅常数 λ"""
        return self.lam

    @property
    def shear_modulus(self) -> float:
        """获取剪切模量 μ"""
        return self.mu

    @property
    def bulk_modulus(self) -> float:
        """获取体积模量 K"""
        return self.calculate_bulk_modulus()
    
    @property
    def hypothesis(self) -> str:
        """获取平面假设"""
        return self.hypo

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
        kwargs = bm.context(self.D)
        D = bm.tensor(self.D[None, None, ...], **kwargs)

        return D
    
    def strain_matrix(self, dof_priority: bool, 
                    gphi: TensorLike, 
                    shear_order: List[str]=['yz', 'xz', 'xy'],
                    # shear_order: List[str]=['xy', 'yz', 'xz'],
                    # shear_order: List[str]=['xy', 'xz', 'yz'], # Abaqus 顺序
                    correction: Optional[str] = None,  # 'None', 'BBar'
                    cm: TensorLike = None, ws: TensorLike = None, detJ: TensorLike = None) -> TensorLike:
        '''
        Constructs the strain-displacement matrix B for the material \n
            based on the gradient of the shape functions.
        B = [∂Ni/∂x   0       0    ]
            [0        ∂Ni/∂y  0    ]
            [0        0       ∂Ni/∂z]
            [0        ∂Ni/∂z  ∂Ni/∂y]
            [∂Ni/∂z   0       ∂Ni/∂x]
            [∂Ni/∂y   ∂Ni/∂x  0     ]

        B = [∂Ni/∂x   0       0    ]
            [0        ∂Ni/∂y  0    ]
            [0        0       ∂Ni/∂z]
            [∂Ni/∂y   ∂Ni/∂x  0    ]
            [∂Ni/∂z   0       ∂Ni/∂x]
            [0        ∂Ni/∂z  ∂Ni/∂y]

        B = [∂Ni/∂x   0       0    ]
            [0        ∂Ni/∂y  0    ]
            [0        0       ∂Ni/∂z]
            [∂Ni/∂y   ∂Ni/∂x  0    ]
            [0        ∂Ni/∂z  ∂Ni/∂y]
            [∂Ni/∂z   0       ∂Ni/∂x]

        Parameters:
            dof_priority (bool): A flag that determines the ordering of DOFs.
                                If True, the priority is given to the first dimension of degrees of freedom.
            gphi - (NC, NQ, LDOF, GD).
            shear_order (List[str], optional): Specifies the order of shear strain components for GD=3.
                                           Valid options are permutations of {'xy', 'yz', 'xz'}.
        
        Returns:
            TensorLike: The strain-displacement matrix `B`, which is a tensor with shape:
                        - For 2D problems (GD=2): (NC, NQ, 3, TLDOF)
                        - For 3D problems (GD=3): (NC, NQ, 6, TLDOF)
        '''
        ldof, GD = gphi.shape[-2:]
        if dof_priority:
            indices = flatten_indices((ldof, GD), (1, 0))
        else:
            indices = flatten_indices((ldof, GD), (0, 1))
        if correction == 'BBar':
            if any(param is None for param in (cm, ws, detJ)):  
                raise ValueError("BBar correction requires cm, ws, and detJ parameters")
            normal_B = self._normal_strain_bbar(gphi, cm, ws, detJ, indices)
        else:
            normal_B = self._normal_strain(gphi, indices)
        
        shear_B = self._shear_strain(gphi, indices, shear_order)

        B = bm.concat([normal_B, shear_B], axis=-2)

        return B
    
    def _normal_strain(self, gphi: TensorLike, 
                    indices: TensorLike, *, 
                    out: Optional[TensorLike]=None) -> TensorLike:
        """Assembly normal strain tensor.

        Parameters:
            gphi - (NC, NQ, LDOF, GD).
            indices - (LDOF, GD): Indices of DoF components in the flattened DoF, shaped .
            out - (TensorLike | None, optional): Output tensor. Defaults to None.

        Returns:
            out - Normal strain shaped (NC, NQ, GD, GD*LDOF): 
        """
        kwargs = bm.context(gphi)
        ldof, GD = gphi.shape[-2:]
        new_shape = gphi.shape[:-2] + (GD, GD*ldof) # (NC, NQ, GD, GD*LDOF)

        if out is None:
            out = bm.zeros(new_shape, **kwargs)
        else:
            if out.shape != new_shape:
                raise ValueError(f'out.shape={out.shape} != {new_shape}')

        for i in range(GD):
            out = bm.set_at(out, (..., i, indices[:, i]), gphi[..., :, i])

        return out
    
    def _normal_strain_bbar(self, gphi: TensorLike,
                        cm, ws, detJ,
                        indices: TensorLike, *,
                        out: Optional[TensorLike] = None) -> TensorLike:
        """Assembly normal strain tensor with B-Bar correction.

        Parameters:
            gphi - (NC, NQ, LDOF, GD).
            indices (TensorLike): Indices of DoF components in the flattened DoF shaped (LDOF, GD).
            out (TensorLike | None, optional): Output tensor. Defaults to None.

        Returns:
            out: B-Bar corrected normal strain shaped (NC, NQ, GD, GD*LDOF).
        """
        kwargs = bm.context(gphi)
        ldof, GD = gphi.shape[-2:]
        new_shape = gphi.shape[:-2] + (GD, GD * ldof)  # (NC, NQ, GD, GD*ldof)

        if out is None:
            out = bm.zeros(new_shape, **kwargs)
        else:
            if out.shape != new_shape:
                raise ValueError(f'out.shape={out.shape} != {new_shape}')

        average_gphi = bm.einsum('cqid, cq, q -> cid', gphi, detJ, ws) / (3 * cm[:, None, None])  # (NC, LDOF, GD)
        for i in range(GD):
            for j in range(GD):
                if i == j:
                    corrected_phi = (2.0 / 3.0) * gphi[..., :, i] + average_gphi[..., None,  :, i] # (NC, NQ, LDOF)
                else:  
                    corrected_phi = (-1.0 / 3.0) * gphi[..., :, j] + average_gphi[..., None, :, j]  # (NC, NQ, LDOF)

                out = bm.set_at(out, (..., i, indices[:, j]), corrected_phi)

        return out

    
    def _shear_strain(self, gphi: TensorLike, 
                    indices: TensorLike, 
                    shear_order: List[str], *,
                    out: Optional[TensorLike]=None) -> TensorLike:
        """Assembly shear strain tensor.

        Parameters:
            gphi - (NC, NQ, LDOF, GD).\n
            indices (bool, optional): Indices of DoF components in the flattened DoF, shaped (LDOF, GD).\n
            out (TensorLike | None, optional): Output tensor. Defaults to None.

        Returns:
            out - Shear strain shaped (NC, NQ, NNZ, GD*LDOF) where NNZ = (GD + (GD+1))//2: .
        """
        kwargs = bm.context(gphi)
        ldof, GD = gphi.shape[-2:]
        if GD < 2:
            raise ValueError(f"The shear strain requires GD >= 2, but GD = {GD}")
        NNZ = (GD * (GD-1))//2    # 剪切应变分量的数量
        new_shape = gphi.shape[:-2] + (NNZ, GD*ldof) # (NC, NQ, NNZ, GD*LDOF)

        if GD == 2:
            shear_indices = [(0, 1)]  # Corresponds to 'xy'
        elif GD == 3:
            valid_pairs = {'xy', 'yz', 'xz'}
            if not set(shear_order).issubset(valid_pairs):
                raise ValueError(f"Invalid shear_order: {shear_order}. Valid options are {valid_pairs}")

            index_map = {
                'xy': (0, 1),
                'yz': (1, 2),
                'xz': (2, 0),
            }
            shear_indices = [index_map[pair] for pair in shear_order]
        else:
            raise ValueError(f"GD={GD} is not supported")

        if out is None:
            out = bm.zeros(new_shape, **kwargs)
        else:
            if out.shape != new_shape:
                raise ValueError(f'out.shape={out.shape} != {new_shape}')

        for cursor, (i, j) in enumerate(shear_indices):
            out = bm.set_at(out, (..., cursor, indices[:, i]), gphi[..., :, j])
            out = bm.set_at(out, (..., cursor, indices[:, j]), gphi[..., :, i])

        return out
    

    

