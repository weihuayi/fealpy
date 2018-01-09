import numpy as np

class FiniteElementFunction(np.ndarray):
    def __new__(cls, V):
        self = V.array().view(cls)
        self.V = V
        return self

    def __call__(self, bc, cellidx=None):
        V = self.V
        return V.value(self, bc, cellidx=cellidx)

    def value(self, bc, cellidx=None):
        V = self.V
        return V.value(self, bc, cellidx=cellidx)

    def grad_value(self, bc, cellidx=None):
        V = self.V
        return V.grad_value(self, bc, cellidx=cellidx)
        
    def div_value(self, bc, cellidx=None):
        V = self.V
        return V.div_value(self, bc, cellidx=cellidx)

    def hessian_value(self, bc, cellidx=None):
        V = self.V
        return V.hessian_value(self, bc, cellidx=cellidx)
