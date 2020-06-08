from sympy import * 
from sympy.abc import x, y


class StokesModel2d:
    """
    -div(nu varepsilon(u)) - nabla p
    """
    def __init__(self, nu, u, p):
        self.nu = nu
        self.u = u
        self.p = p
        self.source()

    def source(self):
        J = self.u.jacobian([x, y])
        J = (J + J.transpose())/2
        f = Matrix([0, 0])
        f[0] = -diff(J[0, 0], x) - diff(J[0, 1], y) - diff(self.p, x)
        f[1] = -diff(J[1, 0], x) - diff(J[1, 1], y) - diff(self.p, y)
        self.f = f
        self.J = J

    def show(self):
        print("nu:", self.nu)
        print("J(x, y):", self.J)
        print("u(x, y):", self.u)
        print("p(x, y):", self.p)
        print("f(x, y):", self.f)
        

if __name__ == "__main__":
    print("Model 0:")
    u = Matrix([sin(pi*x)*cos(pi*y), -cos(pi*x)*sin(pi*y)])
    p = 1/(y**2 + 1) - pi/4
    nu = 1
    model = StokesModel2d(nu, u, p)
    model.show()

    print("Model 1:")
    u = Matrix([-cos(x)**2*cos(y)*sin(y)/2, cos(y)**2*cos(x)*sin(x)/2])
    p = sin(x) - sin(y)
    nu = 1
    model = StokesModel2d(nu, u, p)
    model.show()


    print("Model 2:")
    u = Matrix([2*pi*cos(pi*y)*sin(pi*x)**2*sin(pi*y),
        -2*pi*cos(pi*x)*sin(pi*x)*sin(pi*y)**2])
    p = 0
    nu = 1
    model = StokesModel2d(nu, u, p)
    model.show()

    print("Model 3:")
    u = Matrix([2*pi*cos(pi*y)*sin(pi*x)**2*sin(pi*y),
        -2*pi*cos(pi*x)*sin(pi*x)*sin(pi*y)**2])
    p = sin(x) - sin(y)
    nu = 1
    model = StokesModel2d(nu, u, p)
    model.show()

    print("Model 4:")
    u = Matrix([2*(x**3-x)**2*(y**3-y)*(3*y**2-1),
        -2*(x**3-x)*(3*x**2-1)*(y**3-y)**2])
    p = 1/(1+x**2) - pi/4
    nu = 1
    model = StokesModel2d(nu, u, p)
    model.show()

    print("Model 6:")
    u = Matrix([2*(x**3-x)**2*(y**3-y)*(3*y**2-1),
        -2*(x**3-x)*(3*x**2-1)*(y**3-y)**2])
    p = 0 
    nu = 1
    model = StokesModel2d(nu, u, p)
    model.show()

    print("Model 7:")
    u = Matrix([y**2, x**2])
    p = 0 
    nu = 1
    model = StokesModel2d(nu, u, p)
    model.show()
