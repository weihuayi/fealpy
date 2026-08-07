from typing import Union, Callable, Dict
from .base import BaseEquation

CoefType = Union[int, float, Callable]

class StationarySpecificDissipationRate(BaseEquation):
    def __init__(self, pde, init_variables = False):
        super().__init__(pde)
        self._coefs = {
            'convection': 1.0,      # 对流项系数
            'diffusion': 1.0,       # 扩散项系数
            'cross_diffusion': 1.0, # 交叉扩散项系数
            'dissipation': 1.0,     # 耗散项系数
            'production': 1.0,      # 外力项系数
        }
        self._variables = { 
            'velocity': None,     # 速度变量
            'pressure': None      # 压力变量
        }
        self.pde = pde
        self.initialize_from_pde(pde)
        
        if pde.is_pressure_boundary() == 0 :
            self.pressure_neumann = True
        else:
            self.pressure_neumann = False 
    
    def initialize_from_pde(self, pde):
        """
        根据 pde 对象初始化系数和变量。

        参数:
            pde: PDE 对象，包含 rho, mu, mu_t, R, velocity, pressure, production 等属性

        处理:
            - 如果有 rho 和 mu，直接使用。
            - 如果只有 R，假设 rho=1，计算 mu=rho/R。
            - 如果两者都没有，使用默认值 rho=1, mu=1。
            - 如果有 production，使用其值，否则默认 0。
            - 设置 velocity 和 pressure 的初始值。
        """
        # 处理物理参数
        rho = pde.rho
        mu = pde.mu
        gamma = pde.gamma
        beta = pde.beta
        sigma_omega = pde.sigma_omega
        sigma_omega2 = pde.sigma_omega2

        # 设置系数 
        self._coefs['convection'] = rho
        self._coefs['dissipation'] = beta * rho
        self._coefs['diffusion'] = mu
        self._coefs['cross_diffusion'] = -2 * rho * sigma_omega2
        self._coefs['production'] = gamma * rho
    
    # 定义属性访问
    @property
    def coefs(self):
        """ 返回方程的系数字典。"""
        return self._coefs
    
    @property
    def coef_convection(self):
        """对流项系数"""
        return self._coefs['convection']
    
    @property   
    def coef_diffusion(self):
        """扩散项系数"""
        return self._coefs['diffusion']

    @property
    def coef_cross_diffusion(self):
        """交叉扩散项系数"""
        return self._coefs['cross_diffusion']

    @property
    def coef_dissipation(self):
        """耗散项系数"""
        return self._coefs['dissipation']

    @property
    def coef_production(self):
        """生成项系数"""
        return self._coefs['production']
    
    def set_coefficient(
        self,
        term: str,
        value: CoefType
    ) -> None:
        """
        设置方程项的系数。
        
        参数:
            term: 方程项名称（'mass', 'inertia', 'viscosity', 'pressure'）
            value: 系数值，必须是标量（int/float）或可调用对象（callable）
        
        异常:
            ValueError: 如果值不是标量或可调用对象
            KeyError: 如果方程项名称不存在
        """
        if term not in self._coefs:
            raise KeyError(f"未知方程项: {term}。可选: {list(self._coefs.keys())}")
        
        if not (isinstance(value, (int, float)) or callable(value)):
            raise ValueError("系数必须是标量（int/float）或可调用对象（callable）")
        
        self._coefs[term] = value 
    
    def __str__(self) -> str:
        """返回所有方程项系数和变量的字符串表示"""
        terms_str = "\n".join(
            f"Term[{term}]: {'Callable' if callable(coeff) else coeff}"
            for term, coeff in self._coefs.items()
        )
        
        variables_str = "\n".join(
            f"Variable[{name}]: {type(value).__name__ if value is not None else 'None'}"
            for name, value in self._variables.items()
        )
        
        return f"=== StationarySpecificDissipationRate ===\n" \
               f"coefficients:\n{terms_str}\n\n" \
               f"Variables:\n{variables_str}"
    
    def set_variable(self, name: str, value) -> None:
        """
        设置求解变量（速度或压力）

        Args:
            name: 变量名（'velocity' 或 'pressure'）
            value: 变量值

        Raises:
            KeyError: 如果变量名不存在
        """
        if name not in self._variables:
            raise KeyError(f"Invalid variable '{name}'. Valid variables: {list(self._variables.keys())}")
        self._variables[name] = value

    def set_coefs(self, **kwargs) -> None:
        """
        批量设置方程项系数
        
        Example:
            ns.set_coefs(viscosity=0.1, production=lambda x: x[0])
        """
        for term, value in kwargs.items():
            self.set_coefficient(term, value)

    def set_variables(self, **kwargs) -> None:
        """
        批量设置求解变量
        
        Example:
            ns.set_variables(velocity=u, pressure=p)
        """
        for name, value in kwargs.items():
            self.set_variable(name, value)


    