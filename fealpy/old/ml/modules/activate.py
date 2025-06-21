"""
Activation Functions
"""

from torch import Tensor, exp, sin, cos, tanh, abs
import scipy.special as sp
from torch.nn import Module


def sech(input: Tensor):
    ab = abs(input)
    return 2 * exp(-ab) / (exp(-2 * ab) + 1)


class Activation(Module):
    """
    Base class for activation function module.

    @note: works on singleton mode.
    """
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def dn(self, x: Tensor, order: int) -> Tensor:
        """
        @brief order-specified derivative
        """
        if order == 0:
            return self.forward(x)
        elif order >= 1:
            fn = getattr(self, 'd'+str(order), None)
            if fn is None:
                raise NotImplementedError(f"{order}-order derivative has not been"
                                          f"implemented in {self.__class__.__name__}.")
            else:
                return fn(x)
        else:
            raise ValueError(f"The order can not be negative.")

    def d1(self, p: Tensor) -> Tensor:
        """
        @brief 1-order derivative
        """
        raise NotImplementedError

    def d2(self, p: Tensor) -> Tensor:
        """
        @brief 2-order derivative
        """
        raise NotImplementedError

    def d3(self, p: Tensor) -> Tensor:
        """
        @brief 3-order derivative
        """
        raise NotImplementedError

    def d4(self, p: Tensor) -> Tensor:
        """
        @brief 4-order derivative
        """
        raise NotImplementedError


class Sin(Activation):
    def forward(self, p: Tensor):
        return sin(p)

    def dn(self, x: Tensor, order: int) -> Tensor:
        a, b = divmod(order, 2)
        if b == 0:
            return (-1)**a * sin(x)
        else:
            return (-1)**a * cos(x)

    def d1(self, p: Tensor):
        return cos(p)

    def d2(self, p: Tensor):
        return -sin(p)

    def d3(self, p: Tensor):
        return -cos(p)

    def d4(self, p: Tensor):
        return sin(p)


class Cos(Activation):
    def forward(self, p: Tensor):
        return cos(p)

    def dn(self, x: Tensor, order: int) -> Tensor:
        a, b = divmod(order, 2)
        if b == 0:
            return (-1)**a * cos(x)
        else:
            return -(-1)**a * sin(x)

    def d1(self, p: Tensor):
        return -sin(p)

    def d2(self, p: Tensor):
        return -cos(p)

    def d3(self, p: Tensor):
        return sin(p)

    def d4(self, p: Tensor):
        return cos(p)


class Tanh(Activation):
    def forward(self, p: Tensor):
        return tanh(p)

    def d1(self, p: Tensor):
        return sech(p) ** 2

    def d2(self, p: Tensor):
        return -2 * tanh(p) * sech(p)**2

    def d3(self, p: Tensor):
        return 4 * tanh(p)**2 * sech(p)**2 - 2 * sech(p)**4

    def d4(self, p: Tensor):
        return 16 * tanh(p) * sech(p)**4 - 8 * tanh(p)**3 * sech(p)**2


class Besselj0(Activation):
    def forward(self, p: Tensor):
        return sp.jn(0, p)

    def d1(self, p: Tensor):
        return -sp.jn(1, p)

    def d2(self, p: Tensor):
        return 1/2 * (-sp.jn(0, p) + sp.jn(2, p))

    def d3(self, p: Tensor):
        return 1/4 * (3 * sp.jn(1, p) - sp.jn(3, p))

    def d4(self, p: Tensor):
        return 1/8 * (3 * sp.jn(0, p) - 4 * sp.jn(3, p) + sp.jn(4, p))
