import numpy as np

class FiniteElementFunction(np.ndarray):
    def __new__(cls, V):
        self = V.array().view(cls)
        self.V = V
        return self

    def value(self, bc):
        V = self.V
        return V.value(self, bc)

    def grad_value(self, bc):
        V = self.V
        return V.grad_value(self,bc)
        
    def div_value(self, bc):
        V = self.V
        return V.div_value(self, bc)

    def hessian_value(self, bc):
        V = self.V
        return V.hessian_value(self, bc)
