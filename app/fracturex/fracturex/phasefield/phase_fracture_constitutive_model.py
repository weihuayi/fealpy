import numpy as np
from typing import Optional

from fealpy.experimental.typing import TensorLike
from fealpy.experimental.backend import backend_manager as bm

#from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.experimental.material.elastic_material import LinearElasticMaterial

class PhaseFractureConstitutiveModel(LinearElasticMaterial):
    """
    The class of phase field fracture constitutive model.
    """
    def __init__(self, 
                 E: Optional[float]=None,
                 nu: Optional[float]=None,
                 lam: Optional[float]=None,
                 mu: Optional[float]=None,
                 method: Optional[str]='hybrid',
                 q: Optional[int]=None,
                 hypo: Optional[str]='plane_strain',
                 ) -> None:

        if lam is not None and mu is not None:
            self.E = mu * (3*lam + 2*mu) / (lam + mu)
            self.nu = lam / (2 * (lam + mu))
            self.lam = lam
            self.mu = mu
        elif E is not None and nu is not None:
            self.lam = nu * E / ((1 + nu) * (1 - 2 * nu))
            self.mu = E / (2 * (1 + nu))
            self.E = E
            self.nu = nu
        else:
            raise ValueError("Either (lam, mu) or (E, nu) should be provided.")
        self.q = q if q is not None else 2
        self.method = method
        if hypo not in ["plane_strain", "3D"]:
            raise ValueError("hypo should be either 'plane_strain' or '3D'")
        
        super().__init__(name="MaterialProperties", elastic_modulus=self.E,
                         poisson_ratio=self.nu, hypo=hypo)

        self.hypo = hypo   

    def energy_degradation_function(self, d: TensorLike) -> TensorLike:
        """
        Compute the energy degradation function.
        Attention: The energy degradation function is a scalar function, and the input damage field is a scalar field.
        TODO: The energy degradation function can have other forms.
        Parameters
        ----------
        d : TensorLike
            The damage field.

        Returns
        -------
        TensorLike
            The energy degradation function.
        """
        q = self.q
        mesh = d.space.mesh
        qf = mesh.quadrature_formula(q, 'cell')
        bc, ws = qf.get_quadrature_points_and_weights() 
        eps = 1e-10
        gd = (1 - d(bc))**2 + eps
        return gd

    def effective_stress(self, strain = None) -> TensorLike:
        """
        Compute the effective stress tensor, which is the stress tensor without the damage effect.

        Parameters
        ----------
        u : TensorLike
            The displacement field.
        strain : TensorLike 
            The strain tensor.
        Returns
        -------
        TensorLike
            The effective stress tensor.
        """
        lam = self.lam
        mu = self.mu
        trace_e = np.trace(strain, axis1=-2, axis2=-1)
        I = bm.eye(strain.shape[-1])
        stress = lam * trace_e[..., None, None] * I + 2 * mu * strain
        return stress
       
    def stress(self, strain, d: TensorLike) -> TensorLike:
        """
        Compute the fracture stress tensor.
        """
        method = self.method
        gd = self.energy_degradation_function(d)
        if method == 'hybrid':
            stress = self.effective_stress(strain=strain) * gd[..., None, None]
        else:
            raise ValueError("The method of stress computation is not supported.")
        return stress


    def tangent_stiffness(self, d: TensorLike) -> TensorLike:
        """
        Compute the tangent stiffness tensor.
        """
        method = self.method
        gd = self.energy_degradation_function(d)
        if method == 'hybrid':
            base_D = super().elastic_matrix()
            D = base_D * gd[..., None, None]
        else:
            raise ValueError("The method of tangent stiffness computation is not supported.")
        return D

    def maximum_historical_strain_field(self, u: TensorLike, H) -> TensorLike:
        """
        Compute the maximum historical strain field.
        """
        strain = self.strain(u)
        phip, _ = self.strain_energy_density_decomposition(strain)
        H[:] = np.fmax(H, phip)
        return H

    def strain_energy_density_decomposition(self, s: TensorLike):
        """
        @brief Choose diffient positive and negative decomposition of strain energy density
        """
        method = self.method
        if method == 'spectral':
            return self.spectral_decomposition(s)
        elif method == 'hybrid':
            return self.spectral_decomposition(s)
        else:
            raise ValueError("The method of strain energy density decomposition is not supported.")

    def spectral_decomposition(self, s: TensorLike):
        """
        @brief Strain energy density decomposition from Miehe Spectral
        decomposition method.
        """

        lam = self.lam
        mu = self.mu

        # 应变正负分解
        sp, sm = self.strain_pm_eig_decomposition(s)

        ts = np.trace(s, axis1=-2, axis2=-1)
        tp, tm = self.macaulay_operation(ts)
        tsp = np.trace(sp**2, axis1=-2, axis2=-1)
        tsm = np.trace(sm**2, axis1=-2, axis2=-1)

        phi_p = lam * tp ** 2 / 2.0 + mu * tsp
        phi_m = lam * tm ** 2 / 2.0 + mu * tsm
        return phi_p, phi_m

    def deviatoric_decomposition(self, s: TensorLike):
        pass
    
    def strain_pm_eig_decomposition(self, s: TensorLike):
        """
        @brief Decomposition of Positive and Negative Characteristics of Strain.
        varespilon_{\pm} = \sum_{a=0}^{GD-1} <varespilon_a>_{\pm} n_a \otimes n_a
        varespilon_a is the a-th eigenvalue of strain tensor.
        n_a is the a-th eigenvector of strain tensor.
        
        @param[in] s strain，（NC, NQ, GD, GD）
        """
        w, v = bm.linalg.eigh(s) # w 特征值, v 特征向量
        p, m = self.macaulay_operation(w)

        sp = bm.zeros_like(s)
        sm = bm.zeros_like(s)
        
        GD = s.shape[-1]
        for i in range(GD):
            n0 = v[..., i]  # (NC, NQ, GD)
            n1 = p[..., i, None] * n0  # (NC, NQ, GD)
            sp += n1[..., None] * n0[..., None, :]

            n1 = m[..., i, None] * n0
            sm += n1[..., None] * n0[..., None, :]

        return sp, sm

    
    def macaulay_operation(self, alpha):
        """
        @brief Macaulay operation
        """
        val = bm.abs(alpha)
        p = (alpha + val) / 2.0
        m = (alpha - val) / 2.0
        return p, m

    def heaviside(self, x):
        """
        @brief
        """
        val = bm.zeros_like(x)
        val[x > 1e-13] = 1
        val[bm.abs(x) < 1e-13] = 0.5
        val[x < -1e-13] = 0
        return val
    

