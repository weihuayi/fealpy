from sympy import *

from fealpy.symcom import SimplexElementBasis 

h = Rational(1, 2)

a, b, c = symbols('a, b, c')
t0, t1, t2 = symbols('t0, t1, t2')
x0, x1, x2 = symbols('x0, x1, x2')
y0, y1, y2 = symbols('y0, y1, y2')
p = 4
GD = 2
space = SimplexElementBasis(GD)

l = space.l
print(l)
phi = space.bernstein_basis(p)

f = a*phi[1] + b * phi[3] + c * phi[6]

ft = diff(f, l[0])*t0 + diff(f, l[1])*t1 

eq0 = Eq( f.subs({l[0]:h, l[1]:h}), 1)
eq1 = Eq(ft.subs({l[0]:1, l[1]:0}), 0)
eq2 = Eq(ft.subs({l[0]:0, l[1]:1}), 0) 
solution = solve((eq0, eq1, eq2), (a, b, c))
print(solution)


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

