#!/usr/bin/env python3
# 
from fealpy.symcom import MonomialSpace2d

space = MonomialSpace2d(p=2)
phi = space.basis()
H =  space.integrate()
J = space.jacobian()
G, B = space.construct_G()
print("B0:", B[0])
print("B1:", B[1])
