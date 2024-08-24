from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import grad


class KernelFunctionBase(ABC):
    def __init__(self, h: float):
        self.h_derivative = 1.0/h

    def __call__(self, r):
        return self.value(r)

    @abstractmethod
    def value(self, r):
        pass

    def grad_value(self, r):
        return grad(self.value)(r)

class QuinticKernel(KernelFunctionBase): 

    def __init__(self,h,dim=3):
        self.h_derivative = 1.0/h
        self.constant = 3.0
        self.cutoff_radius = self.constant * h 
        
        if dim == 1:
            self.alpha = 1.0/120.0 * self.h_derivative
        elif dim == 2:
            self.alpha = 7.0/478.0/jnp.pi * self.h_derivative**2
        elif dim == 3:
            self.alpha = 3.0/359.0/jnp.pi * self.h_derivative**3

    def value(self, r):
        q = r * self.h_derivative
        q0 = jnp.maximum(0.0,1.0-q)
        q1 = jnp.maximum(0.0,2.0-q)
        q2 = jnp.maximum(0.0,3.0-q)
        return self.alpha * (q2**5 - 6.0 * q1**5 + 15.0 * q0**5)

class CubicSplineKernel(KernelFunctionBase):

    def __init__(self,h,dim=3):
        self.h_derivative = 1.0/h
        self.constant = 2.0
        self.cutoff_radius = self.constant * h 

        if dim == 1:
            self.alpha = 2.0/3.0 * self.h_derivative
        elif dim == 2:
            self.alpha = 10.0/7.0/jnp.pi * self.h_derivative**2
        elif dim == 3:
            self.alpha = 1.0/jnp.pi * self.h_derivative**3

    def value(self,r):
        q = r * self.h_derivative
        q0 = jnp.maximum(0.0,1.0-q)
        q1 = jnp.maximum(0.0,2.0-q)
        q2 = 1.0 - q + q**2
        return self.alpha * ((1.0/4.0) * q1**3 - q2 * q0)

class QuadraticKernel(KernelFunctionBase):

    def __init__(self,h,dim=3):
        self.h_derivative = 1.0/h
        self.constant = 2.0
        self.cutoff_radius = self.constant * h 

        if dim == 1:
            self.alpha = 2.0/3.0 * self.h_derivative
        elif dim == 2:
            self.alpha = 2.0/jnp.pi * self.h_derivative**2
        elif dim == 3:
            self.alpha = 5.0/4.0/jnp.pi * self.h_derivative**3
    
    def value(self,r):
        q = r * self.h_derivative
        q0 = jnp.maximum(0.0,2.0-q)
        return self.alpha * (3.0/16.0 * q0**2)

class WendlandC2Kernel(KernelFunctionBase):
    
    def __init__(self,h,dim=3):
        self.h_derivative = 1.0/h
        self.constant = 2.0
        self.cutoff_radius = self.constant * h
        self.dim = dim

        if dim == 1:
            self.alpha = 5.0/8.0 * self.h_derivative
        elif dim == 2:
            self.alpha = 7.0/4.0/jnp.pi * self.h_derivative**2
        elif dim == 3:
            self.alpha = 21.0/16.0/jnp.pi * self.h_derivative**3

    def value(self,r):
        if self.dim == 1:
            q = r * self.h_derivative
            q0 = jnp.maximum(0.0,1.0-0.5*q)
            q1 = 1.5 * q + 1.0
            return self.alpha * (q0**3 * q1)
        else:
            q = r * self.h_derivative
            q0 = jnp.maximum(0.0,1.0-0.5*q)
            q1 = 2.0 * q + 1.0
            return self.alpha * (q0**4 * q1)
