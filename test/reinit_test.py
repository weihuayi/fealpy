import numpy as np

from fealpy.mesh.level_set_function import dcircle
from fealpy.mesh.StructureQuadMesh import StructureQuadMesh 

def circle(p, cxy, r):
    x = p[:, 0]
    y = p[:, 1]
    return  (x - cxy[0])**2 + (y - cxy[1])**2 - r**2

def sign(phi):
    eps = 1e-12
    sign = np.sign(phi)
    sign[np.abs(phi) < eps] = 0
    return sign

def HG(mesh, phi):

box = [-1, 1, -1, 1]
cxy = (0.0, 0.0)
r = 0.5
interface = lambda p: dcircle(p, cxy, r)

nx = 10
ny = 10

mesh = StructureQuadMesh(box, nx, ny)

phi = interface(mesh.point)
signPhi = sign(phi)



