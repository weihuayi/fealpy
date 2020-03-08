
from sympy import *
from sympy.abc import x, y

alpha, mu, lam, C1, C2, r, theta = symbols('alpha, mu, lam, C1, C2, r, theta')
rr = sqrt(x**2 + y**2)
th = atan2(y, x)

u = Matrix([[0], [0]])
u[0, 0]  = -(alpha + 1)*cos((alpha+1)*th)
u[0, 0] += (C2 - alpha - 1)*C1*cos((alpha -1)*th)
u[0, 0] /= 2*mu
u[0, 0] *= rr**alpha

u[1, 0]  = (alpha + 1)*sin((alpha+1)*th)
u[1, 0] += (C2 + alpha - 1)*C1*sin((alpha -1)*th)
u[1, 0] /= 2*mu
u[1, 0] *= rr**alpha

du = u.jacobian((x, y))
print('du:\n', du.subs({x**2+y**2:r**2, atan2(y, x):theta}))

strain = (du.transpose() + du)/2
print('strain:\n', strain)

stress = 2*mu*strain + lam*strain.trace()*eye(2)
print('stress:\n', stress)

f0 = -diff(stress[0, 0], x) - diff(stress[0, 1], y)
f1 = -diff(stress[1, 0], x) - diff(stress[1, 1], y)
print('f0', f0.simplify())
print('f1', f1.simplify())

