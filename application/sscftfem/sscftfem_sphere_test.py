import sys

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh.level_set_function import Sphere
from SSCFTFEMModel import SSCFTFEMModel,SSCFTParameter

import plotly.offline as py
import plotly.figure_factory as FF

m = int(sys.argv[1])

# Sphere Type
if m == 1:
    surface = Sphere()
    mesh =  surface.init_mesh()
    mesh.uniform_refine(n=4, surface=surface)


option = SSCFTParameter()
option.maxit = 5000
scft = SSCFTFEMModel(surface, mesh, option, p=1, p0=1)


femspace = scft.femspace

Ndof = femspace.number_of_global_dofs()
ipoint = femspace.interpolation_points()
cell = mesh.ds.cell
chiN = option.chiAB * option.Ndeg

scftinfo = '''
========== model parameters ==========
chiN:%f
fA:%f
======= Discretization Points ========
point:%d
cell:%d

'''%(chiN, option.fA, ipoint.shape[0],cell.shape[0])
print(scftinfo)


# define fields
fields = np.zeros((Ndof, 2))
pi = np.pi
theta = np.arctan2(ipoint[:,1], ipoint[:,0])
theta = (theta >= 0)*theta + (theta <0)*(theta + 2*pi)             
fields[:, 1] = chiN*np.sin(5*theta)
option.fields = fields
scft.initialize()
scft.find_saddle_point(datafile='spheredata', file_path='./')

