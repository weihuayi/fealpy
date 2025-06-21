from .base import BaseEquation
from typing import Union, Callable, Dict
CoefType = Union[int, float, Callable]

class CahnHilliard(BaseEquation):
    def __init__(self, pde=None, init_variables=True):
        super().__init__(pde)
        self._coefs = {
            'mobility': 1,  # 时间导数项系数
            'convection': 1,        # 对流项系数
            'interface': 1,       # 界面项系数
            'free_energy': 1      # 自由能导数项系数
        }
        self._variables = {
            'phi': None,          # 相场变量
            'mu': None            # 化学势
        }
        if init_variables:
            if value is None:
                raise ValueError("Value cannot be None")
            self.initialize_from_pde(pde)

    def initialize_from_pde(self, pde):
        """
        根据 pde 对象初始化系数和变量。

        参数:
            pde: PDE 对象，包含 convection, epsilon, init_phi, init_mu 等属性

        处理:
            - 如果有 convection，直接使用。
            - 如果有 epsilon，计算 interface 系数为 epsilon^2。
            - 如果没有提供，使用默认值 convection=1, epsilon=1。
            - 设置 phi 和 mu 的初始值。
        """
        # 处理物理参数
        convection = getattr(pde, 'convection', 1.0)
        epsilon = getattr(pde, 'epsilon', 1.0)

        # 设置系数
        self._coefs['mobility'] = 1.0
        self._coefs['convection'] = convection
        self._coefs['interface'] = epsilon ** 2
        self._coefs['free_energy'] = 1.0

        # 设置变量
        self._variables['phi'] = getattr(pde, 'init_phi', None)
        self._variables['mu'] = getattr(pde, 'init_mu', None)

    # 定义变量访问
    @property
    def variables(self):
        """变量字典"""
        return self._variables

    @property
    def phi(self):
        """相场变量"""
        return self._variables['phi']

    @property
    def mu(self):
        """化学势"""
        return self._variables['mu']

    # 定义属性访问
    @property
    def coefs(self):
        """系数字典"""
        return self._coefs

    @property
    def coef_mobility(self) -> float | Callable:
        """流动性系数"""
        return self._coefs['mobility']

    @property
    def coef_convection(self) -> CoefType:
        """对流项系数项"""
        return self._coefs['convection']

    @property
    def coef_interface(self) -> CoefType:
        """界面项系数"""
        return self._coefs['interface']

    @property
    def coef_free_energy(self) -> CoefType:
        """自由能导数项系数"""
        return self._coefs['free_energy']

    def set_coefficient(
        self,
        term: str,
        value: CoefType
    ) -> None:
        """
        设置方程项的系数。

        参数:
            term: 方程项名称（'mobility', 'convection', 'interface', 'free_energy'）
            value: 系数值，必须是标量（int/float）或可调用对象（callable）

        异常:
            ValueError: 如果值不是标量或可调用对象
            KeyError: 如果方程项名称不存在
        """
        if term not in self._coefs:
            raise KeyError(f"未知方程项: {term}。可选: {list(self._coefs.keys())}")

        #if not (isinstance(value, (int, float)) or callable(value)):
        #    raise ValueError("系数必须是标量（int/float）或可调用对象（callable）")

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

        return f"=== CahnHilliard ===\n" \
               f"coefficients:\n{terms_str}\n\n" \
               f"Variables:\n{variables_str}"

    def set_variable(self, name: str, value) -> None:
        """
        设置求解变量（相场或化学势）

        Args:
            name: 变量名（'phi' 或 'mu'）
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
            ch.set_coefs(convection=0.1, interface=0.01)
        """
        for term, value in kwargs.items():
            self.set_coefficient(term, value)

    def set_variables(self, **kwargs) -> None:
        """
        批量设置求解变量

        Example:
            ch.set_variables(phi=phi_field, mu=mu_field)
        """
        for name, value in kwargs.items():
            self.set_variable(name, value)
