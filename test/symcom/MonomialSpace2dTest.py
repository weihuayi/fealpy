#!/usr/bin/env python3
# 
from fealpy.symcom import MonomialSpace2d

space = MonomialSpace2d(p=3)
phi = space.basis()
H =  space.integrate()
J = space.jacobian()
G, B = space.construct_G()
print("G00:", G[0])
print("G11:", G[1])
print("G01:", G[2])
print("B0:", B[0])
print("B1:", B[1])
