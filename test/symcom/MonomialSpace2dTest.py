#!/usr/bin/env python3
# 
from fealpy.symcom import MonomialSpace2d

space = MonomialSpace2d(p=2)
phi = space.basis()
H =  space.integrate()
print(H)
