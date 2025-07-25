from builtins import float, str
from typing import Optional, Tuple, List

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from fealpy.material.elastic_material import LinearElasticMaterial


class ElastoplasticMaterial(LinearElasticMaterial):
    '''
    ElastoplasticMaterial represents a linear elastoplastic material with optional linear hardening.
    
    This class models materials that exhibit both elastic and plastic behavior according to the von Mises yield criterion, 
    with support for linear isotropic hardening. 
    It extends LinearElasticMaterial by adding yield stress and hardening modulus, 
    and provides methods for computing the elastoplastic tangent matrix and related quantities. 
    Suitable for finite element analysis of elastoplastic solids.
    
    Parameters
        name : str
            Name of the material.
        yield_stress : float
            Initial yield stress (scalar), i.e., the stress at which plastic deformation begins.
        hardening_modulus : float, optional, default=0.0
            Linear hardening modulus (H'), representing the slope of the yield surface in stress-plastic strain space.
        **kwargs
            Additional keyword arguments passed to the base LinearElasticMaterial class (e.g., elastic_modulus, poisson_ratio).
            
    Attributes
        yield_stress : float
            Initial yield stress of the material.
        hardening_modulus : float
            Linear hardening modulus (H'). If zero, perfect plasticity is assumed.
        is_hardening : bool
            Whether the material considers hardening effects (True if hardening_modulus > 0).
    Methods
        df_dsigma(stress)
            Compute the derivative of the yield function with respect to the Cauchy stress (for von Mises criterion).
        deviatoric_stress(stress)
            Compute the deviatoric (traceless) part of the stress tensor in Voigt notation.
        elastico_plastic_matrix(stress=None, df_dsigma=None)
            Compute the elastoplastic tangent matrix D^p at the current stress state.
        elastico_plastic_hardening_matrix(stress=None, plastic_strain=None, df_dsigma=None)
            Compute the elastoplastic tangent matrix D^p considering linear hardening.
    Notes
        - The class assumes small strain theory and uses Voigt notation for stress and strain tensors.
        - Only linear (isotropic) hardening is supported; nonlinear hardening requires further extension.
        - The elastoplastic tangent operator is computed using the return mapping algorithm for von Mises plasticity.
    '''
    def __init__(self, 
                 name: str,
                 yield_stress: float,          
                 hardening_modulus: float = 0.0, 
                 **kwargs):
        """
        Initialize the elastoplastic material with yield stress and optional hardening modulus.
        
        Parameters
            name : str
                Name of the material.
            yield_stress : float
                Initial yield stress (scalar), i.e., the stress at which plastic deformation begins.
            hardening_modulus : float, optional, default=0.0
                Linear hardening modulus (H'), representing the slope of the yield surface in stress-plastic strain space.
            **kwargs
                Additional keyword arguments passed to the base LinearElasticMaterial class (e.g., elastic_modulus
                , poisson_ratio).
        """
        super().__init__(name, **kwargs)
        self.yield_stress = yield_stress
        self.hardening_modulus = hardening_modulus  # 考虑线性硬化参数

    def df_dsigma(self, stress: TensorLike) -> TensorLike:
        """
        Compute the derivative of the yield function with respect to the Cauchy stress (von Mises criterion).

        Parameters
            stress: Cauchy stress tensor (Voigt notation)
            - 3D: (..., 6) [sigma_11, sigma_22, sigma_33, sigma_12, sigma_23, sigma_13]
            - 2D: (..., 3) [sigma_11, sigma_22, sigma_12]

        Returns
            df_dsigma: Derivative tensor (same shape as stress)
        """
        # 计算偏应力 (deviatoric stress)
        s = self.deviatoric_stress(stress)
        
        # 计算等效应力 (von Mises应力)
        if self.hypo == "3D":
            J2 = 0.5 * (s[...,0]**2 + s[...,1]**2 + s[...,2]**2) + \
                 s[...,3]**2 + s[...,4]**2 + s[...,5]**2
        else:  # 2D情况
            J2 = (s[...,0]**2 + s[...,1]**2 - s[...,0]*s[...,1])/3 + \
                 s[...,2]**2
            
        J2 = bm.maximum(J2, 1e-10)  # 防止除零
        df = 1.5 * s / bm.sqrt(3*J2)[..., None]
        return df

    def deviatoric_stress(self, stress: TensorLike) -> TensorLike:
        """
        Compute the deviatoric (traceless) part of the stress tensor in Voigt notation.
        """
        s = stress.copy()
        if self.hypo == "3D":
            mean = (stress[...,0] + stress[...,1] + stress[...,2])/3
            s[...,0] -= mean
            s[...,1] -= mean
            s[...,2] -= mean
        else:  # 平面应力/应变
            mean = (stress[...,0] + stress[...,1])/3
            s[...,0] -= mean
            s[...,1] -= mean
            if self.hypo == "plane_strain":
                s[...,2] -= mean  # σ33分量
        return s

    def elastico_plastic_matrix(self, 
                     stress: Optional[TensorLike] = None, 
                     df_dsigma: Optional[TensorLike] = None) -> TensorLike:
        """
        Compute the elastoplastic tangent matrix D^p.

        Parameters
            stress: Current stress tensor (used to automatically compute df_dsigma)
            df_dsigma: Directly provide the derivative tensor (optional)
        
        Returns
            Dp: Elastoplastic tangent matrix, with the same shape as the base class D matrix, but computed independently for each integration point
        """
        # 获取弹性矩阵
        De = super().elastic_matrix()  # (1,1,N,N)
        
        # 获取导数张量
        if df_dsigma is None:
            if stress is None:
                raise ValueError("需要提供stress或df_dsigma")
            df = self.df_dsigma(stress)  # (NC,NQ,N)
        else:
            df = df_dsigma

        # 扩展De到匹配df的维度
        De_exp = bm.broadcast_to(De, df.shape[:-1] + De.shape[-2:])  # (NC,NQ,N,N)
        
        # 计算 a = De : df
        a = bm.einsum('...ij,...j->...i', De_exp, df)  # (NC,NQ,N)
        
        # 计算分母 H = df:De:df + H'
        H = bm.einsum('...i,...i->...', df, a) + self.hardening_modulus  # (NC,NQ)
        H = bm.maximum(H, 1e-10)  # 避免除零
        
        # 计算塑性修正项
        numerator = bm.einsum('...i,...j->...ij', a, a)  # (NC,NQ,N,N)
        Dp = De_exp - numerator / H[..., None, None]      # (NC,NQ,N,N)
        
        return Dp

    @property
    def is_hardening(self) -> bool:
        """Whether hardening effects are considered"""
        return self.hardening_modulus > 0
    
    def elastico_plastic_hardening_matrix(self, 
                     stress: Optional[TensorLike] = None, 
                     plastic_strain: Optional[TensorLike] = None,
                     df_dsigma: Optional[TensorLike] = None) -> TensorLike:
        """
        Compute the elastoplastic hardening matrix D^p.

        Parameters
            stress: Current stress tensor (used to automatically compute df_dsigma)
            plastic_strain: Current plastic strain tensor (used to compute hardening effects)
            df_dsigma: Directly provide the derivative tensor (optional)

        Returns
            Dp: Elastoplastic hardening matrix, with the same shape as the base class D matrix, but computed independently for each integration point
        """
        # 获取弹性矩阵
        De = super().elastic_matrix()  # (1,1,N,N)
        
        # 获取导数张量
        if df_dsigma is None:
            if stress is None:
                raise ValueError("需要提供stress或df_dsigma")
            df = self.df_dsigma(stress)  # (NC,NQ,N)
        else:
            df = df_dsigma

        # 扩展De到匹配df的维度
        De_exp = bm.broadcast_to(De, df.shape[:-1] + De.shape[-2:])  # (NC,NQ,N,N)
        
        # 计算当前屈服应力（考虑线性硬化）#TODO: 未考虑非线性硬化
        if plastic_strain is not None:
            # σ_y = σ_y0 + H' * ε_p
            current_yield = self.yield_stress + self.hardening_modulus * plastic_strain
        else:
            current_yield = self.yield_stress
        
        # 计算 a = De : df
        a = bm.einsum('...ij,...j->...i', De_exp, df)  # (NC,NQ,N)
        
        # 计算分母 H = df:De:df + H'
        H = bm.einsum('...i,...i->...', df, a) + self.hardening_modulus  # (NC,NQ)
        H = bm.maximum(H, 1e-10)  # 避免除零
        
        # 计算塑性修正项
        numerator = bm.einsum('...i,...j->...ij', a, a)  # (NC,NQ,N,N)
        Dp = De_exp - numerator / H[..., None, None]      # (NC,NQ,N,N)
        
        return Dp