from builtins import float, str
from typing import Optional, Tuple, List

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from fealpy.material.elastic_material import LinearElasticMaterial


class ElastoplasticMaterial(LinearElasticMaterial):
    """
    ElastoplasticMaterial represents an isotropic elastoplastic material supporting the von Mises yield criterion and isotropic linear hardening.
    
    This class is suitable for small deformation theory and uses the Voigt notation, making it applicable for finite element elastoplastic analysis.

    Parameters:
        name : str
            Material name
        yield_stress : float
            Initial yield stress (stress at which material begins to plastically deform)
        hardening_modulus : float, optional, default=0.0
            Linear hardening modulus (H'), slope of the yield surface with respect to equivalent plastic strain
        **kwargs
            Additional parameters passed to the base class LinearElasticMaterial (e.g., elastic_modulus, poisson_ratio)

    Attributes:
        yield_stress : float
            Initial yield stress (MPa)
        hardening_modulus : float
            Linear hardening modulus; if zero, ideal elastoplastic behavior
        is_hardening : bool
            Whether hardening is considered (hardening_modulus > 0)

    Methods:
        df_dsigma(stress)
            Compute the derivative of the von Mises yield function with respect to Cauchy stress
        deviatoric_stress(stress)
            Compute the deviatoric part of the stress (removing volumetric component)
        elastico_plastic_matrix(stress=None, df_dsigma=None)
            Compute the elastoplastic tangent matrix D^p for the current stress state
        elastico_plastic_hardening_matrix(stress=None, plastic_strain=None, df_dsigma=None)
            Compute the elastoplastic tangent matrix D^p considering linear hardening
    """
    def __init__(self, 
                 name: str,
                 yield_stress: float,          
                 hardening_modulus: float = 0.0, 
                 **kwargs):
        """
        Initialize the elastoplastic material with yield stress and optional hardening modulus.
        
        Parameters:
            name : str
                Material name
            yield_stress : float
                Initial yield stress (MPa)
            hardening_modulus : float, optional, default=0.0
                Linear hardening modulus (H'), slope of the yield surface with respect to equivalent plastic strain
            **kwargs
                Additional parameters passed to the base class LinearElasticMaterial (e.g., elastic_modulus, poisson_ratio)
        """
        super().__init__(name, **kwargs)
        self.yield_stress = yield_stress
        self.hardening_modulus = hardening_modulus  # Linear hardening modulus parameter

    def yield_function(self, stress: TensorLike, alpha: TensorLike) -> TensorLike:
        """
        compute the von Mises yield function value

        Parameters:
            stress : TensorLike
                Cauchy stress tensor (NC, NQ, N)
            alpha : TensorLike
                Equivalent plastic strain (NC, NQ)
                
        Returns:
            f :  TensorLike
        """
        s = self.deviatoric_stress(stress)
        norm_s = bm.sqrt(bm.sum(s**2, axis=-1))
        seq = bm.sqrt(3.0 / 2.0) * norm_s
        f = seq - (self.yield_stress + self.hardening_modulus * alpha)
        return f

    def plastic_normal(self, stress: TensorLike) -> TensorLike:
        """
        Return the derivative (normal vector) of the von Mises yield function with respect to stress.

        N = sqrt(3/2) * s / ||s||, where s is the deviatoric stress.

        Parameters:
            stress: Current stress tensor (NC, NQ, N)

        Returns:
            N: Unit normal vector to the yield surface (NC, NQ, N)
        """
        s = self.deviatoric_stress(stress)
        norm_s = bm.sqrt(bm.sum(s**2, axis=-1))
        norm_s = bm.maximum(norm_s, 1e-10)  
        N = bm.sqrt(3.0 / 2.0) * s / norm_s[..., None]
        return N


    def df_dsigma(self, stress: TensorLike) -> TensorLike:
        """
        Compute the derivative of the yield function with respect to Cauchy stress (von Mises criterion).
        Compute the derivative of the yield function with respect to Cauchy stress (von Mises criterion).

        Parameters:
            stress: Cauchy stress tensor (Voigt notation)
            - 3D: (..., 6) [sigma_11, sigma_22, sigma_33, sigma_12, sigma_23, sigma_13]
            - 2D: (..., 3) [sigma_11, sigma_22, sigma_12]

        Returns:
            df_dsigma: Derivative tensor (same shape as stress)
        """
        # Compute deviatoric stress
        s = self.deviatoric_stress(stress)
        
        # Compute equivalent stress (von Mises stress)
        if self.hypo == "3D":
            J2 = 0.5 * (s[...,0]**2 + s[...,1]**2 + s[...,2]**2) + \
             s[...,3]**2 + s[...,4]**2 + s[...,5]**2
        else:  # 2D case
            J2 = (s[...,0]**2 + s[...,1]**2 - s[...,0]*s[...,1])/3 + \
             s[...,2]**2
            
        J2 = bm.maximum(J2, 1e-10)  # Prevent division by zero
        df = 1.5 * s / bm.sqrt(3*J2)[..., None]
        return df

    def deviatoric_stress(self, stress: TensorLike) -> TensorLike:
        """
        Compute the deviatoric stress tensor.
        This removes the hydrostatic part of the stress tensor, leaving only the deviatoric component.
        
        Parameters:
            stress: Cauchy stress tensor (NC, NQ, N) in Voigt

        Returns:
            deviatoric_stress: Deviatoric stress tensor (NC, NQ, N)
        Compute the deviatoric stress tensor.
        This removes the hydrostatic part of the stress tensor, leaving only the deviatoric component.
        """
        s = stress.copy()
        if self.hypo == "3D":
            mean = (stress[...,0] + stress[...,1] + stress[...,2])/3
            s[...,0] -= mean
            s[...,1] -= mean
            s[...,2] -= mean
        else:  # 2D case
            mean = (stress[...,0] + stress[...,1])/3
            s[...,0] -= mean
            s[...,1] -= mean
            if self.hypo == "plane_strain":
                s[...,2] -= mean  
                s[...,2] -= mean  
        return s
    
    def elastico_plastic_matrix_isotropic(self, 
                     stress: Optional[TensorLike] = None) -> TensorLike:
        """
        Compute the elastoplastic tangent matrix D^p.

        Parameters:
            stress: Current stress tensor (used to automatically compute df_dsigma)
            df_dsigma: Directly provide the derivative tensor (optional)
            
        Returns:
            Dp: Elastoplastic tangent matrix, same shape as the base class D matrix, computed independently for each integration point
        """
        E = self.elastic_modulus
        nu = self.poisson_ratio
        H = self.hardening_modulus
        G = E / (2 * (1 + nu))
        De = self.elastic_matrix()  #  (1,1,N,N)
        N = self.plastic_normal(stress) 
        value = bm.einsum('...i,...j->...ij', N, N)  # (NC,NQ,N,N)
        coef = 6 * G**2 / (3 * G + H)
        D_ep = De - coef * value  # (NC,NQ,N,N)
        
        return D_ep
        
        

    def elastico_plastic_matrix(self, 
                     stress: Optional[TensorLike] = None, 
                     df_dsigma: Optional[TensorLike] = None) -> TensorLike:
        """
        Compute the elastoplastic tangent matrix D^p.

        Parameters:
            stress: Current stress tensor (used to automatically compute df_dsigma)
            df_dsigma: Directly provide the derivative tensor (optional)

        Returns:
            Dp: Elastoplastic tangent matrix, same shape as the base class D matrix, computed independently for each integration point
        """
        De = super().elastic_matrix()  # (1,1,N,N)
        
        
        if df_dsigma is None:
            if stress is None:
                raise ValueError("需要提供stress或df_dsigma")
            df = self.plastic_normal(stress)  # (NC,NQ,N)
            df = self.plastic_normal(stress)  # (NC,NQ,N)
        else:
            df = df_dsigma

        De_exp = bm.broadcast_to(De, df.shape[:-1] + De.shape[-2:])  # (NC,NQ,N,N)
        
        a = bm.einsum('...ij,...j->...i', De_exp, df)  # (NC,NQ,N)
        
        H = bm.einsum('...i,...i->...', df, a) + self.hardening_modulus  # (NC,NQ)
        H = bm.maximum(H, 1e-10)  
        
        numerator = bm.einsum('...i,...j->...ij', a, a)  # (NC,NQ,N,N)
        Dp = De_exp - numerator / H[..., None, None]      # (NC,NQ,N,N)
        
        return Dp

    @property
    def is_hardening(self) -> bool:
        """
        Check if the material has hardening behavior.
        
        Returns:
            bool: True if hardening_modulus > 0, False otherwise.
        """
        return self.hardening_modulus > 0

        
    def material_point_update(self, delta_strain, strain_pl_n, strain_e_n):
        '''
        Perform the elastoplastic constitutive update for a material point.
        
        This method computes the updated stress, plastic strain, 
        and equivalent plastic strain based on the current strain increment and previous state variables.
        
        Parameters:
            delta_strain : TensorLike
                Incremental strain tensor (NC, NQ, 3) at the current time step.
            strain_pl_n : TensorLike
                Previous plastic strain tensor (NC, NQ, 3).
            strain_e_n : TensorLike
                Previous equivalent plastic strain (scalar) at the current time step (NC, NQ).
                
            delta_strain : TensorLike
                Incremental strain tensor (NC, NQ, 3) at the current time step.
            strain_pl_n : TensorLike
                Previous plastic strain tensor (NC, NQ, 3).
            strain_e_n : TensorLike
                Previous equivalent plastic strain (scalar) at the current time step (NC, NQ).
                
        Returns:
            Tuple[TensorLike, TensorLike, TensorLike, TensorLike, TensorLike]:
                - sigma_np1: Updated stress tensor (NC, NQ, 3).
                - strain_pl_n1: Updated plastic strain tensor (NC, NQ, 3).  
                - strain_e_n1: Updated equivalent plastic strain (scalar) at the current time step (NC, NQ).
                - Ctang: Tangent stiffness matrix (NC, NQ, 3, 3).
                - is_plastic: Boolean mask indicating whether plastic deformation occurred (NC, NQ).
        '''
        E = self.elastic_modulus
        nu = self.poisson_ratio
        H = self.hardening_modulus
        G = E / (2 * (1 + nu))
        
        De = self.elastic_matrix()  # (NC, NQ, 3, 3)

        strain_total = delta_strain + strain_pl_n  # (NC, NQ, 3)
        sigma_trial = bm.einsum('...ij,...j->...i', De, strain_total)  # (NC, NQ, 3)
        s_trial = self.deviatoric_stress(sigma_trial)  # (NC, NQ, 3)
        stress_trial_e = bm.sqrt(3.0 / 2.0) * bm.sqrt(bm.sum(s_trial ** 2, axis=-1))  # (NC, NQ)

        f_trial = self.yield_function(sigma_trial, strain_e_n)  # (NC, NQ)
        is_plastic = f_trial > 0  # (NC, NQ) 

        gamma = f_trial / (3 * G + H)  # (NC, NQ)
        stress_trial_e = bm.maximum(stress_trial_e, 1e-12)
        n = s_trial / (2 / 3 * stress_trial_e[..., None])  # (NC, NQ, 3)

        sigma_np1 = bm.where(
            is_plastic[..., None],
            sigma_trial - 2 * G * gamma[..., None] * n,
            sigma_trial
        )

        strain_pl_n1 = bm.where(
            is_plastic[..., None],
            strain_pl_n + gamma[..., None] * n,
            strain_pl_n
        )

        strain_e_n1 = bm.where(
            is_plastic,
            strain_e_n + gamma,
            strain_e_n
        )

        NC = sigma_np1.shape[0]
        NQ = sigma_np1.shape[1]
        Ctang = bm.broadcast_to(De, (NC, NQ, 3, 3))  # (NC, NQ, 3, 3)
       

        # Compute Ctang for plastic points
        if bm.any(is_plastic):
            sigma_pl = bm.where(is_plastic[..., None], sigma_np1, 0.0)  # (NC, NQ, 3)
            df = self.plastic_normal(sigma_pl)                          # (NC, NQ, 3)
            #Ctang_plastic = self.elastico_plastic_matrix(sigma_pl, df)  # (NC, NQ, 3, 3)
            Ctang_plastic = self.elastico_plastic_matrix_isotropic(sigma_pl)  # (NC, NQ, 3, 3)
            Ctang = bm.where(is_plastic[..., None, None], Ctang_plastic, De)  # (NC, NQ, 3, 3)


        return sigma_np1, strain_pl_n1, strain_e_n1, Ctang, is_plastic
