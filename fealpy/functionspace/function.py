import numpy as np

class FiniteElementFunction(np.ndarray):
    def __new__(cls, V, dim=None, array=None):
        if array is None:
            self = V.array(dim=dim).view(cls)
        else:
            self = array.view(cls)
            print(array)
        self.V = V
        return self

    def index(self, i):
        return FiniteElementFunction(self.V, array=self[:, i])

    def __add__(self, y):
        a = self.view(np.ndarray)
        return FiniteElementFunction(self.V, array=a+y)

    def __sub__(self, y):
        a = self.view(np.ndarray)
        return FiniteElementFunction(self.V, array=a-y)

    def __mul__(self, y):
        a = self.view(np.ndarray)
        return FiniteElementFunction(self.V, array=a*y)

    def __div__(self, y):
        a = self.view(np.ndarray)
        return FiniteElementFunction(self.V, array=a/y)

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
