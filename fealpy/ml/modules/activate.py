"""
Activation Functions
"""

from torch import Tensor, exp, sin, cos, tanh, abs
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
