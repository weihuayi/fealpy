from .base import BaseEquation
from typing import Union, Callable, Dict
CoefType = Union[int, float, Callable]

class StationaryStokes(BaseEquation):
    def __init__(self, pde):
        super().__init__(pde)
        self._coefs = {
            'pressure': 1,        # 压力项系数
            'viscosity': 1,       # 粘性项系数
            'body_force': 1      # 外力项系数
        }
        self.pde = pde
        self.initialize_from_pde(pde) 
    
    def initialize_from_pde(self, pde):
        """
        根据 pde 对象初始化系数和变量。

        参数:
            pde: PDE 对象，包含 rho, mu, H, R, velocity, pressure, reaction 等属性

        处理:
            - 如果有 rho, mu 和 H 直接使用。
            - 如果只有 R，假设 rho=1，计算 mu=rho/R。
            - 如果两者都没有，使用默认值 rho=1, mu=1。
            - 设置 velocity 和 pressure 的初始值。
        """
        # 处理物理参数
        if hasattr(pde, 'mu'):
            mu = pde.mu
        else:
            mu = 1.0

        # 设置系数 
        self._coefs['pressure'] = 1
        self._coefs['viscosity'] = mu
        self._coefs['body_force'] = getattr(pde, 'source', 0)


    # 定义属性访问
    @property
    def coefs(self):
        """系数字典"""
        return self._coefs

    @property
    def coef_viscosity(self) -> CoefType:
        """粘性项系数"""
        return self._coefs['viscosity']

    @property
    def coef_pressure(self) -> CoefType:
        """压力项系数"""
        return self._coefs['pressure']
    
    @property
    def coef_body_force(self) -> CoefType:
        """外力项系数"""
        return self._coefs['body_force']
    
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
        
        return f"=== Stokes ===\n" \
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
            ns.set_coefs(viscosity=0.1, body_force=lambda x: x[0])
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


    
