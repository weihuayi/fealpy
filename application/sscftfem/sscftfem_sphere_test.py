import sys

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh.level_set_function import Sphere
from SSCFTFEMModel import SSCFTFEMModel,SSCFTParameter

m = int(sys.argv[1])
initType = int(sys.argv[2]) 

# Sphere Type
if m == 1:
    surface = Sphere()
    mesh =  surface.init_mesh()
    mesh.uniform_refine(n=0, surface=surface)

# ready sscftmodel Parameter
point = mesh.point # point of mesh
Ndof = point.shape[0] # number of point on mesh

option = SSCFTParameter()
chiN = option.chiAB * option.Ndeg                                      

# define fields 
fields = np.zeros((Ndof, 2))

if option.fieldType is 'fieldmu':
    if initType == 1:
        fields[:12, 1] = 1
    elif initType == 2:
        fields[:, 0] = chiN*(-1 + 2*np.random.rand(1, Ndof))                
        fields[:, 1] = chiN*(-1 + 2*np.random.rand(1, Ndof))
    elif initType == 3:
        pi = np.pi
        theta = np.arctan2(point[:,1],point[:,0])
        theta = (theta >= 0)*theta + (theta <0)*(theta + 2*pi)             
        fields[:, 1] = chiN*np.sin(5*theta)
elif option.fieldType is 'fieldw':
    if initType == 4:
        pass
scft = SSCFTFEMModel(surface,mesh,option,fields,p=1,p0=1)
scft.initialize()
scft.update_propagator()
scft.find_saddle_point()


