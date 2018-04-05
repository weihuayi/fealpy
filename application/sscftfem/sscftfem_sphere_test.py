import sys

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh.level_set_function import Sphere
from SSCFTFEMModel import SSCFTFEMModel,SSCFTParameter

import plotly.offline as py
import plotly.figure_factory as FF

m = int(sys.argv[1])
model_pmt = int(sys.argv[2])
fieldsType = int(sys.argv[3])


# Sphere Type
if m == 1:
    surface = Sphere()
    mesh =  surface.init_mesh()
    mesh.uniform_refine(n=5, surface=surface)

option = SSCFTParameter()

# model parameters
if model_pmt == 1:
    option.fA = 0.2
    option.chiN = 0.25
elif model_pmt ==2:
    option.fA = 0.5
    option.chiN = 0.15

option.pdemethod = 'CN'
option.maxit = 5000
option.showstep = 50
  

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
node:%d
cell:%d

'''%(chiN, option.fA, ipoint.shape[0],cell.shape[0])
print(scftinfo)


# define fields
fields = np.zeros((Ndof, 2))

if fieldsType == 1:
    fields[:12, 1] = 1
elif fieldsType == 2:
    fields[:, 0] = chiN*(-1 + 2*np.random.rand(1, Ndof))
    fields[:, 1] = chiN*(-1 + 2*np.random.rand(1, Ndof))
elif fieldsType == 3:
    k = 5
    pi = np.pi
    theta = np.arctan2(ipoint[:,1], ipoint[:,0])
    theta = (theta >= 0)*theta + (theta <0)*(theta + 2*pi)             
    fields[:, 1] = chiN*np.sin(k*theta)
elif fieldsType == 4:
    k  = 5
    pi = np.pi
    theta = np.arctan2(ipoint[:,1], ipoint[:,0])
    theta = (theta >= 0)*theta + (theta <0)*(theta + 2*pi)
    phi = np.arctan(ipoint[:, 3], ipoint[:, 2])
    phi = (phi >= 0)*phi + (phi <0)*(phi + pi)     
    fields[:, 1] = chiN*np.sin(k*theta+k*phi)
elif fieldsType == 5:
    pi = np.pi
    fields[:, 1] = chiN*(np.sin(pi*ipoint[:,0]))
elif fieldsType == 6:
    pi = np.pi
    fields[:, 1] = chiN*(np.sin(pi*ipoint[:,0]))
   
option.fields = fields

scft.initialize()
scft.find_saddle_point(datafile='spheredata', file_path='./')

