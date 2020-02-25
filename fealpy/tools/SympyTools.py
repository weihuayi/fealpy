from sympy import *
import sympy as sym
import sympy.printing as printing


class SympyTools():
    def __init__(self):
        pass

    def symbols(self, var):
        return symbols(var)

    def model(self, u, var, returnlatex=False):
        from sympy.tensor.array import derive_by_array, tensorproduct
        m = len(var)
        gradient = Matrix(derive_by_array(u, var))
        length = sqrt(trace(Matrix(tensorproduct(gradient, gradient)).reshape(m, m)))
        n = gradient/length
        div = Matrix(derive_by_array(n, var)).reshape(m, m)
        div_n = trace(div)
#        div_n = sym.simplify(div_n)
        hessian = Matrix(derive_by_array(gradient, var)).reshape(m, m)
        laplace = trace(hessian)
        laplace = sym.simplify(laplace)
        val = {'grad': gradient, 'Hessian': hessian, 'Laplace': laplace,
                'unit_normal':n, 'div_unit_normal':div_n}
        if returnlatex is False:
            return val
        else:
            print('grad:\n', printing.latex(val['grad']))
            print('Hessian:\n', printing.latex(val['Hessian']))
            print('Laplace:\n', printing.latex(val['Laplace']))
            print('unit_normal:\n', printing.latex(val['unit_normal']))
            print('div_unit_normal:\n', printing.latex(val['div_unit_normal']))


