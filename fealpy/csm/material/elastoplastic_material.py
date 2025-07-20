from fealpy.backend import backend_manager as bm

from builtins import float, str

from fealpy.typing import TensorLike
from typing import Optional, Tuple, List
from fealpy.material.elastic_material import LinearElasticMaterial


class ElastoplasticMaterial(LinearElasticMaterial):
    """
    ElastoplasticMaterial 表示支持各向同性材料、von Mises 屈服准则和各向同性线性硬化的弹塑性材料。
    该类适用于小变形理论，采用 Voigt 表示法，适合有限元弹塑性分析。
    支持特性:
    - 各向同性弹性材料
    - von Mises 屈服准则
    - 各向同性线性硬化（hardening_modulus > 0 时）
    参数
    name : str
        材料名称
    yield_stress : float
        初始屈服应力（材料进入塑性变形的应力）
    hardening_modulus : float, optional, default=0.0
        线性硬化模量（H'），为屈服面随等效塑性应变增长的斜率
    **kwargs
        传递给基类 LinearElasticMaterial 的其他参数（如 elastic_modulus, poisson_ratio）
    属性
    yield_stress : float
        初始屈服应力
    hardening_modulus : float
        线性硬化模量，若为0则为理想弹塑性
    is_hardening : bool
        是否考虑硬化效应（hardening_modulus > 0）
    方法
    df_dsigma(stress)
        计算 von Mises 屈服函数对柯西应力的导数
    deviatoric_stress(stress)
        计算应力的偏应力部分（去除体积分量）
    elastico_plastic_matrix(stress=None, df_dsigma=None)
        计算当前应力状态下的弹塑性切线矩阵 D^p
    elastico_plastic_hardening_matrix(stress=None, plastic_strain=None, df_dsigma=None)
        计算考虑线性硬化的弹塑性切线矩阵 D^p
    注意
    - 仅支持各向同性材料、von Mises 屈服和各向同性线性硬化
    - 非线性硬化、各向异性材料等需进一步扩展
    - 切线算子采用 von Mises 塑性理论的返回映射算法推导
    示例
    >>> mat = ElastoplasticMaterial(name="Steel", yield_stress=250.0, elastic_modulus=210e3, poisson_ratio=0.3)
    >>> stress = np.array([260.0, 240.0, 0.0])
    >>> df = mat.df_dsigma(stress)
    >>> Dp = mat.elastico_plastic_matrix(stress=stress)
    """
    def __init__(self, 
                 name: str,
                 yield_stress: float,          # 初始屈服应力
                 hardening_modulus: float = 0.0, # 硬化模量 (H')
                 **kwargs):
        """
        Args:
            yield_stress: 初始屈服应力 (标量)
            hardening_modulus: 硬化模量 (H' = dsigma_y/dε_p)
            **kwargs: 基类参数 (elastic_modulus, poisson_ratio等)
        """
        super().__init__(name, **kwargs)
        self.yield_stress = yield_stress
        self.hardening_modulus = hardening_modulus  # 考虑线性硬化参数

    def df_dsigma(self, stress: TensorLike) -> TensorLike:
        """
        计算屈服函数对柯西应力的导数 (von Mises准则)
        
        Args:
            stress: 柯西应力张量 (Voigt表示法)
                - 3D: (..., 6) [sigma_11, sigma_22, sigma_33, sigma_12, sigma_23, sigma_13]
                - 2D: (..., 3) [sigma_11, sigma_22, sigma_12]
        
        Returns:
            df_dsigma: 导数张量 (与stress同维度)
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
        计算偏应力张量 (Voigt表示法)
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
        计算弹塑性矩阵 D^p

        Args:
            stress: 当前应力张量 (用于自动计算df_dsigma)
            df_dsigma: 直接提供导数张量 (可选)
        
        Returns:
            Dp: 弹塑性矩阵,形状与基类D矩阵相同,但每个积分点独立计算
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
        """是否考虑硬化效应"""
        return self.hardening_modulus > 0
    
    def material_update(self, 
                     stress: Optional[TensorLike] = None, 
                     plastic_strain: Optional[TensorLike] = None,
                     df_dsigma: Optional[TensorLike] = None) -> TensorLike:
        """
        更新材料状态，计算弹塑性矩阵 D^p
        
        Args:
            stress: 当前应力张量 (用于自动计算df_dsigma)
            plastic_strain: 当前塑性应变张量 (可选)
            df_dsigma: 直接提供导数张量 (可选)
        
        Returns:
            Dp: 弹塑性矩阵
        """
        return self.elastico_plastic_hardening_matrix(stress, plastic_strain, df_dsigma)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    '''
    def elastico_plastic_hardening_matrix(self, 
                     stress: Optional[TensorLike] = None, 
                     plastic_strain: Optional[TensorLike] = None,
                     df_dsigma: Optional[TensorLike] = None) -> TensorLike:
        """
        计算弹塑性矩阵 D^p

        Args:
            stress: 当前应力张量 (用于自动计算df_dsigma)
            df_dsigma: 直接提供导数张量 (可选)
        
        Returns:
            Dp: 弹塑性矩阵,形状与基类D矩阵相同,但每个积分点独立计算
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
        
        # 计算当前屈服应力（考虑线性硬化）#TODO: 未考虑非线性硬化（添加一个self.kaapa）
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
    '''