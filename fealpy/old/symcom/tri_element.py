
from sympy import *

class TriElement:
    def __init__(self):
        self.u = symbols('xi, eta') # 参考坐标
        self.x = symbols('x, y, z') # 实际坐标
        self.l = symbols('lambda_:3') # 重心坐标 

    def reference_shape_function_2(self):
        c12 = Rational(1, 2)
        xi  = self.u[0]
        eta = self.u[1]
        phi = Matrix([
            [2 * (1 - xi - eta) * (Rational(1, 2) - xi - eta)],
            [xi * (2 * xi - 1)],
            [eta * (2 * eta - 1)],
            [4 * xi * eta],
            [4 * eta * (1 - xi - eta)],
            [4 * xi * (1 - xi -eta)]])
        return phi


if __name__ == '__main__':
    elem = TriElement()

    phi = elem.reference_shape_function_2()
    dxi = diff(phi, elem.u[0])
    deta = diff(phi, elem.u[1])

    print(phi)
    print(latex(dxi))
    print(latex(deta))
