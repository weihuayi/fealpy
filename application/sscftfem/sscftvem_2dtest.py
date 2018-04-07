import sys
import numpy as np

from SCFTVEMModel import SCFTVEMModel, SCFTParameter
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh 


n = int(sys.argv[1])  # n is mesh refine
fieldsType = int(sys.argv[2]) # define fields type

box = [-1, 1, -1, 1]
mesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='polygon')

# get_sys_parameter
option = SCFTParameter()
scft = SCFTVEMModel(mesh, option, p=1) 
chiN = option.Ndeg*option.chiAB


vemspace = scft.vemspace
Ndof = vemspace.number_of_global_dofs()

# define fields
fields = np.zeros((Ndof, 2))

if fieldsType == 1:
    fields[:, 0] = chiN*(-1 + 2*np.random.rand(1, Ndof))
    fields[:, 1] = chiN*(-1 + 2*np.random.rand(1, Ndof))
elif fieldsType == 2:
    pi = np.pi
    fields[:, 1] = chiN*(np.sin(pi*node[:,0]))
elif fieldsType == 6:
    pi = np.pi
    fields[:, 1] = chiN*(np.sin(pi*node[:,0]))
   
option.fields = fields

scft.update_propagator()
scft.initialize()
scft.find_saddle_point(datafile='vemflat', file_path='./')

