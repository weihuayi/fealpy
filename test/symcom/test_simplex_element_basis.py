
from sympy import *
from fealpy.symcom import SimplexElementBasis
from fealpy.mesh import TriangleMesh

def subs(exp, s, var=None):
    for key, val in s.items():
        exp = exp.subs(key, val)
        if var is not None:
            exp = exp.subs(Derivative(val, var), Derivative(key, var))
    return exp

def test_hermite_basis():
    GD = 2
    p = 5
    space = SimplexElementBasis(GD)
    phi = space.bernstein_basis(p)
    l = space.l
    a, b, c = symbols('a, b, c')
    x0, x1, x2 = symbols('x0, x1, x2')
    y0, y1, y2 = symbols('y0, y1, y2')

    f0  = a*phi[0] + b*phi[1] + c*phi[2]
    f0x = diff(f0, l[0])*x0 + diff(f0, l[1])*x1 + diff(f0, l[2])*x2
    f0y = diff(f0, l[0])*y0 + diff(f0, l[1])*y1 + diff(f0, l[2])*y2

    eq0 = Eq( f0.subs({l[0]:1, l[1]:0, l[2]:0}), 1)
    eq1 = Eq(f0x.subs({l[0]:1, l[1]:0, l[2]:0}), 0)
    eq2 = Eq(f0y.subs({l[0]:1, l[1]:0, l[2]:0}), 0)
    solution = solve((eq0, eq1, eq2), (a, b, c))
    print(solution)

    eq0 = Eq( f0.subs({l[0]:1, l[1]:0, l[2]:0}), 0)
    eq1 = Eq(f0x.subs({l[0]:1, l[1]:0, l[2]:0}), 1)
    eq2 = Eq(f0y.subs({l[0]:1, l[1]:0, l[2]:0}), 0)
    solution = solve((eq0, eq1, eq2), (a, b, c))
    print(solution)

    eq0 = Eq(f0.subs({l[0]:1,  l[1]:0, l[2]:0}), 0)
    eq1 = Eq(f0x.subs({l[0]:1, l[1]:0, l[2]:0}), 0)
    eq2 = Eq(f0y.subs({l[0]:1, l[1]:0, l[2]:0}), 1)
    solution = solve((eq0, eq1, eq2), (a, b, c))
    print(solution)

if __name__ == "__main__":
    TriangleMesh.show_shape_function(p=3, funtype='B')
    TriangleMesh.show_grad_shape_function(p=3, funtype='B')
    TriangleMesh.show_lattice(p=3)
    test_hermite_basis()

