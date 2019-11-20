from sympy import *

class SympyTools:

    def __init__(self):
        pass

    def symbols(self, s):
        return symbols(s)

    def laplace_model(self, u, var):
        from sympy.tensor.array import derive_by_array
        grad = derive_by_array(u, var)
        hess = derive_by_array(grad, var)
        source = -hess.trace()
        return {'grad': grad, 'Hessian': hess, 'source': source}

