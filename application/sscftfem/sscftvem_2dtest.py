import sys
import numpy as np

from fealpy.functionspace.vem_space import VirtualElementSpace2d 
from SCFTVEMModel import SCFTVEMModel, SCFTParameter
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh 
import matplotlib.pyplot as plt


n = int(sys.argv[1])  # n is mesh refine
fieldsType = int(sys.argv[2]) # define fields type

box = [0, 4, 0, 4]
mesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='polygon')

node = mesh.node
p = 1
vemspace =VirtualElementSpace2d(mesh, p)

# get_sys_parameter
option = SCFTParameter()
option.maxit = 100
chiN = option.Ndeg*option.chiAB

# define fields
Ndof = vemspace.number_of_global_dofs()
fields = np.zeros((Ndof, 2))

if fieldsType == 1:
    fields[:, 0] = chiN*(-1 + 2*np.random.rand(1, Ndof))
    fields[:, 1] = chiN*(-1 + 2*np.random.rand(1, Ndof))
elif fieldsType == 2:
    fields[:, 1] = chiN*(np.sin(5*node[:,0]))
elif fieldsType == 3:
    fields[:, 1] = chiN*(np.sin(5*node[:,1]))
   
option.fields = fields

scft = SCFTVEMModel(vemspace, option) 

scft.initialize()
#scft.update_propagator()
scft.find_saddle_point()
plt.show()

