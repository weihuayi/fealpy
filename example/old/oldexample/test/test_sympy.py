import sympy as sym
import numpy as np
from sympy.vector import CoordSys3D, Del, curl

C = CoordSys3D('C')
x = sym.symbols("x")
y = sym.symbols("y")
z = sym.symbols("z")

f = sym.sin(C.y)*C.i + sym.sin(C.x)*C.j + sym.sin(C.z)*C.k
a = curl(curl(f))
b = a.dot(C.i).subs({C.x:x, C.y:y, C.z:z})
f = sym.lambdify(('x', 'y', 'z'), b, "numpy")
x = np.array([1, 2, 3])

print(f(x[1], x[0], x[2]))
