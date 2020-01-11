from fealpy.symcom import MonomialSpace2d

space = MonomialSpace2d(p=3)
phi = space.basis()
H =  space.integrate()
print(H)
