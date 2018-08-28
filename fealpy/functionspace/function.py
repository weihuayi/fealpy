import numpy as np

class Function(np.ndarray):
    def __new__(cls, space, dim=None, array=None):
        if array is None:
            self = space.array(dim=dim).view(cls)
        else:
            self = array.view(cls)
        self.space = space 
        return self

    def index(self, i):
        return FiniteElementFunction(self.space, array=self[:, i])

    def __call__(self, bc, cellidx=None):
        space = self.space
        return space.value(self, bc, cellidx=cellidx)

    def value(self, bc, cellidx=None):
        space = self.space
        return space.value(self, bc, cellidx=cellidx)

    def grad_value(self, bc, cellidx=None):
        space = self.space
        return space.grad_value(self, bc, cellidx=cellidx)
        
    def div_value(self, bc, cellidx=None):
        space = self.space
        return space.div_value(self, bc, cellidx=cellidx)

    def hessian_value(self, bc, cellidx=None):
        space = self.space
        return space.hessian_value(self, bc, cellidx=cellidx)
