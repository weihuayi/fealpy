import math
from builtins import float, str
from typing import Optional, Tuple, List

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import variantmethod

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
                 hardening_modulus: float = 0.1, 
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
            stress(Tensor): Cauchy stress tensor (NC, NQ, N)
            alpha(Tensor): Equivalent plastic strain (NC, NQ)
                
        Returns:
            f(Tensor): Yield function value (NC, NQ)
        """
        s = self.deviatoric_stress(stress)
        norm_s = bm.sqrt(bm.sum(s**2, axis=-1))
        seq = math.sqrt(3.0 / 2.0) * norm_s
        print(seq.shape, alpha.shape)
        f = seq - (self.yield_stress + self.hardening_modulus * alpha)
        return f

    def plastic_normal(self, stress: TensorLike) -> TensorLike:
        """
        Return the derivative (normal vector) of the von Mises yield function with respect to stress.

        N = sqrt(3/2) * s / ||s||, where s is the deviatoric stress.

        Parameters:
            stress(Tensor): Current stress tensor (NC, NQ, N)

        Returns:
            N(Tensor): Unit normal vector to the yield surface (NC, NQ, N)
        """
        s = self.deviatoric_stress(stress)
        norm_s = bm.sqrt(bm.sum(s**2, axis=-1))
        norm_s = bm.maximum(norm_s, bm.array(1e-10))  
        N = math.sqrt(3.0 / 2.0) * s / norm_s[..., None]
        return N


    def df_dsigma(self, stress: TensorLike) -> TensorLike:
        """
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
    
    def plastic_potential(self, stress: TensorLike) -> TensorLike:
        """
        Von Mises plastic potential (equivalent stress).
        φ(σ) = sqrt(3/2 * s:s) = sqrt(3 * J2)
        
        For associated flow, this is also the yield function.
        
        Parameters:
            stress: (NC, NQ, N) — Cauchy stress tensor in vector form
        
        Returns:
            phi: (NC, NQ) — equivalent stress (scalar per integration point)
        """
        
        # Step 1: Compute deviatoric stress
        s = self.deviatoric_stress(stress)  # (NC, NQ, N)
        
        # Step 2: Compute J2 = 0.5 * s_ij * s_ij
        # 注意：对于工程剪应变格式，剪应力项要乘 0.5 来修正内积
        if s.shape[-1] == 3:  # 2D
            # s = [s_xx, s_yy, s_xy]
            J2 = 0.5 * (s[..., 0]**2 + s[..., 1]**2 + s[..., 0]*s[..., 1]) + 0.5 * s[..., 2]**2
            # 更严谨的方式（推荐）：
            # J2 = 0.5 * (s[..., 0]**2 + s[..., 1]**2 - s[..., 0]*s[..., 1]) + 0.5 * s[..., 2]**2  # 平面应变
        elif s.shape[-1] == 6:  # 3D
            # s = [s_xx, s_yy, s_zz, s_xy, s_yz, s_zx]
            # s:s = s_xx^2 + s_yy^2 + s_zz^2 + 2*(s_xy^2 + s_yz^2 + s_zx^2)
            # But in engineering notation, shear terms are NOT doubled,
            # so we must account for that in inner product:
            normal_part = s[..., 0]**2 + s[..., 1]**2 + s[..., 2]**2
            shear_part = 2.0 * (s[..., 3]**2 + s[..., 4]**2 + s[..., 5]**2)
            J2 = 0.5 * (normal_part + shear_part)
        else:
            raise ValueError(f"Unsupported stress dimension: {s.shape[-1]}")
        
        # Step 3: Equivalent stress = sqrt(3 * J2)
        phi = bm.sqrt(3.0 * J2)  # (NC, NQ)
        
        return phi

    def deviatoric_stress(self, stress: TensorLike) -> TensorLike:
        """
        Compute the deviatoric stress tensor.
        This removes the hydrostatic part of the stress tensor, leaving only the deviatoric component.
        
        Parameters:
            stress(Tensor): Cauchy stress tensor (NC, NQ, N) in Voigt

        Returns:
            deviatoric_stress(Tensor): Deviatoric stress tensor (NC, NQ, N)
        """
        s = bm.copy(stress)
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
        return s
    
    def elastico_plastic_matrix_isotropic(self, 
                     stress: Optional[TensorLike] = None) -> TensorLike:
        """
        Compute the elastoplastic tangent matrix D^p.

        Parameters:
            stress(Tensor): Current stress tensor (used to automatically compute df_dsigma)
            df_dsigma(Tensor): Directly provide the derivative tensor (optional)
            
        Returns:
            Dp(Tensor): Elastoplastic tangent matrix, same shape as the base class D matrix, computed independently for each integration point
        """
        E = self.elastic_modulus
        nu = self.poisson_ratio
        H = self.hardening_modulus
        G = E / (2 * (1 + nu))
        De = self.elastic_matrix()  #  (1,1,N,N)
        N = self.plastic_normal(stress) 
        value = bm.einsum('...i,...j->...ij', N, N)  # (NC,NQ,N,N)
        coef = 4 * G**2 / (3 * G + H)
        D_ep = De - coef * value  # (NC,NQ,N,N)
        
        return D_ep 
    
    def elastico_plastic_matrix_backend(self, 
                            stress: Optional[TensorLike] = None,
                            df_dsigma: Optional[TensorLike] = None) -> TensorLike:
        """
        Compute the elastoplastic tangent matrix D^p.
        
        Automatically uses autograd if backend is PyTorch;
        otherwise uses analytical derivative via plastic_normal().
        """
        # 1. 获取弹性矩阵和后端
        De = super().elastic_matrix()  # (1,1,N,N)
        backend = bm.get_current_backend(logger_msg="in elastico_plastic_matrix")

        # 2. 如果用户直接提供了 df_dsigma，直接使用（跳过自动/解析求导）
        if df_dsigma is not None:
            df = df_dsigma
        else:
            if stress is None:
                raise ValueError("Either stress or df_dsigma must be provided.")
            
            # 3. 根据后端类型选择求导方式
            if backend == 'pytorch':
                import torch
                # 确保 stress 是 torch.Tensor 且可微
                if not isinstance(stress, torch.Tensor):
                    raise TypeError("Stress must be a torch.Tensor when using PyTorch backend.")
                if not stress.requires_grad:
                    stress = stress.clone().detach().requires_grad_(True)

                # 假设 plastic_potential 返回标量势（每个积分点一个值）
                phi = self.plastic_potential(stress)  # shape: (NC, NQ)
                phi_sum = phi.sum()
                df = torch.autograd.grad(phi_sum, stress, create_graph=True)[0]  # (NC, NQ, N)

            else:
                # NumPy / JAX / 其他：使用解析导数
                df = self.plastic_normal(stress)  # (NC, NQ, N)

        # 4. 统一组装流程（与后端无关）
        De_exp = backend.broadcast_to(De, df.shape[:-1] + De.shape[-2:])  # (NC, NQ, N, N)
        a = backend.einsum('...ij,...j->...i', De_exp, df)  # (NC, NQ, N)
        
        H = backend.einsum('...i,...i->...', df, a) + self.hardening_modulus  # (NC, NQ)
        H = backend.maximum(H, backend.array(1e-10))  # 避免除零
        
        numerator = backend.einsum('...i,...j->...ij', a, a)  # (NC, NQ, N, N)
        Dp = De_exp - numerator / H[..., None, None]  # (NC, NQ, N, N)
    
        return Dp
         
    def elastico_plastic_matrix(self, 
                     stress: Optional[TensorLike] = None, 
                     df_dsigma: Optional[TensorLike] = None) -> TensorLike:
        """
        Compute the elastoplastic tangent matrix D^p.

        Parameters:
            stress(Tensor): Current stress tensor (used to automatically compute df_dsigma)
            df_dsigma(Tensor): Directly provide the derivative tensor (optional)

        Returns:
            Dp(Tensor): Elastoplastic tangent matrix, same shape as the base class D matrix, computed independently for each integration point
        """
        De = super().elastic_matrix()  # (1,1,N,N)
        
        
        if df_dsigma is None:
            if stress is None:
                raise ValueError("需要提供stress或df_dsigma")
            df = self.plastic_normal(stress)  # (NC,NQ,N)
        else:
            df = df_dsigma

        De_exp = bm.broadcast_to(De, df.shape[:-1] + De.shape[-2:])  # (NC,NQ,N,N)
        
        a = bm.einsum('...ij,...j->...i', De_exp, df)  # (NC,NQ,N)
        
        H = bm.einsum('...i,...i->...', df, a) + self.hardening_modulus  # (NC,NQ)
        H = bm.maximum(H, bm.array(1e-10))  
        
        numerator = bm.einsum('...i,...j->...ij', a, a)  # (NC,NQ,N,N)
        Dp = De_exp - numerator / H[..., None, None]      # (NC,NQ,N,N)
        
        return Dp 
         
    @property
    def is_hardening(self) -> bool:
        """
        Check if the material has hardening behavior.
        
        Returns:
            bool(Bool): True if hardening_modulus > 0, False otherwise.
        """
        return self.hardening_modulus > 0

    def material_point_update(self, strain_total, strain_pl_n, strain_e_n):
        '''
        Perform the elastoplastic constitutive update for a material point.
        
        This method computes the updated stress, plastic strain, 
        and equivalent plastic strain based on the current strain increment and previous state variables.
        
        Parameters:
            delta_strain(TensorLike):
                Incremental strain tensor (NC, NQ, 3) at the current time step
            strain_pl_n(TensorLike):
                Previous plastic strain tensor (NC, NQ, 3)
            strain_e_n(TensorLike):
                Previous equivalent plastic strain (scalar) at the current time step (NC, NQ)
                
        Returns:
            sigma_np1(TensorLike):
                Updated Cauchy stress tensor (NC, NQ, 3)
            strain_pl_n1(TensorLike):
                Updated plastic strain tensor (NC, NQ, 3)
            strain_e_n1(TensorLike):
                Updated equivalent plastic strain (scalar) at the current time step (NC, NQ)
            Ctang(TensorLike):
                Elastoplastic tangent matrix (NC, NQ, 3, 3)
            is_plastic(TensorLike):
                Boolean mask indicating whether the point is in the plastic regime (NC, NQ)
        '''
        E = self.elastic_modulus
        nu = self.poisson_ratio
        H = self.hardening_modulus
        G = E / (2 * (1 + nu))
        
        De = self.elastic_matrix()  # (NC, NQ, 3, 3)

        sigma_trial = bm.einsum('...ij,...j->...i', De, strain_total-strain_pl_n)  # (NC, NQ, 3)
        s_trial = self.deviatoric_stress(sigma_trial)  # (NC, NQ, 3)
        stress_trial_e = math.sqrt(3.0 / 2.0) * bm.sqrt(bm.sum(s_trial ** 2, axis=-1))  # (NC, NQ)
        stress_trial_e =  bm.maximum(stress_trial_e, bm.array(1e-12))  # 避免除零错误

        f_trial = self.yield_function(sigma_trial, strain_e_n)  # (NC, NQ)
        is_plastic = f_trial > 0  # (NC, NQ) 

        gamma = f_trial / (3 * G + H)  # (NC, NQ)
        stress_trial_e = bm.maximum(stress_trial_e, bm.array(1e-12))
        # TODO:检查n的数学表达式是否正确
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
        Ctang = bm.broadcast_to(De, (NC, NQ, De.shape[2], De.shape[3]))  # (NC, NQ, 3, 3)
       
        # Compute Ctang for plastic points
        if bm.any(is_plastic):
            sigma_pl = bm.where(is_plastic[..., None], sigma_np1, 0.0)  # (NC, NQ, 3)
            df = self.plastic_normal(sigma_pl)                          # (NC, NQ, 3)
            if  H == 0:
                Ctang_plastic = self.elastico_plastic_matrix_backend(sigma_pl, df)  # (NC, NQ, 3, 3)
            else:
                Ctang_plastic = self.elastico_plastic_matrix_isotropic(sigma_pl)  # (NC, NQ, 3, 3)
            Ctang = bm.where(is_plastic[..., None, None], Ctang_plastic, De)  # (NC, NQ, 3, 3)

        return sigma_np1, strain_pl_n1, strain_e_n1, Ctang, is_plastic


    # TODO: 检查这个很可能是这个的问题
    def material_point_update_ideal(self, strain_total, strain_pl_n, strain_e_n):
        '''
        Perform the elastoplastic constitutive update for a material point.
        
        This method computes the updated stress, plastic strain, 
        and equivalent plastic strain based on the current strain increment and previous state variables.
        
        Parameters:
            delta_strain(TensorLike):
                Incremental strain tensor (NC, NQ, 3) at the current time step
            strain_pl_n(TensorLike):
                Previous plastic strain tensor (NC, NQ, 3)
            strain_e_n(TensorLike):
                Previous equivalent plastic strain (scalar) at the current time step (NC, NQ)
                
        Returns:
            sigma_np1(TensorLike):
                Updated Cauchy stress tensor (NC, NQ, 3)
            strain_pl_n1(TensorLike):
                Updated plastic strain tensor (NC, NQ, 3)
            strain_e_n1(TensorLike):
                Updated equivalent plastic strain (scalar) at the current time step (NC, NQ)
            Ctang(TensorLike):
                Elastoplastic tangent matrix (NC, NQ, 3, 3)
            is_plastic(TensorLike):
                Boolean mask indicating whether the point is in the plastic regime (NC, NQ)
        '''
        E = self.elastic_modulus
        nu = self.poisson_ratio
        H = self.hardening_modulus
        G = E / (2 * (1 + nu))
        
        De = self.elastic_matrix()  # (NC, NQ, 3, 3)

        sigma_trial = bm.einsum('...ij,...j->...i', De, strain_total-strain_pl_n)  # (NC, NQ, 3)
        s_trial = self.deviatoric_stress(sigma_trial)  # (NC, NQ, 3)
        stress_trial_e = bm.sqrt(3.0 / 2.0) * bm.sqrt(bm.sum(s_trial ** 2, axis=-1))  # (NC, NQ)

        f_trial = self.yield_function(sigma_trial, strain_e_n)  # (NC, NQ)
        is_plastic = f_trial > 0  # (NC, NQ) 

        gamma = f_trial / (3 * G + H)  # (NC, NQ)
        stress_trial_e = bm.maximum(stress_trial_e, 1e-12)
        # TODO:检查n的数学表达式是否正确
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
            # 提取塑性点的真实应力（不是 zero-padded 的）
            sigma_pl_valid = sigma_np1[is_plastic]  # shape: (N_plastic, 3)
            df_valid = n  # safe
            De_exp_valid = bm.broadcast_to(De, (sigma_pl_valid.shape[0],) + De.shape[-2:])
            
            a_valid = bm.einsum('...ij,...j->...i', De_exp_valid, df_valid)
            H_eff_valid = bm.einsum('...i,...i->...', df_valid, a_valid) + self.hardening_modulus
            H_eff_valid = bm.maximum(H_eff_valid, 1e-10)
            
            numerator_valid = bm.einsum('...i,...j->...ij', a_valid, a_valid)
            Dp_valid = De_exp_valid - numerator_valid / H_eff_valid[..., None, None]
            
            # 写回 Ctang
            Ctang_plastic = bm.zeros_like(Ctang)
            Ctang_plastic[is_plastic] = Dp_valid
            Ctang = bm.where(is_plastic[..., None, None], Ctang_plastic, Ctang)

        return sigma_np1, strain_pl_n1, strain_e_n1, Ctang, is_plastic
