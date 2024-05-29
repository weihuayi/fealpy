from abc import abstractmethod

import jax.numpy as jnp
from jax import grad


class KernelFunctionBase():
    def __init__(self, h: float):
        pass

    @abstractmethod
    def value(self, r):
        pass

    def grad_value(self, r):
        return grad(self.value)(r)



class Name(KernelFunctionBase): 

    def __init__():
        pass
