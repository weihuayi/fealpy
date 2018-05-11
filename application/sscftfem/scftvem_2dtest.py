import sys
import numpy as np

from fealpy.functionspace.vem_space import VirtualElementSpace2d 
from SCFTVEMModel import SCFTVEMModel, SCFTParameter
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh 


n = int(sys.argv[1])  # n is mesh refine
fieldsType = int(sys.argv[2]) # define fields type

# define squre mesh size
option = SCFTParameter()
option.t = 5*np.pi
box = [0, option.t, 0, option.t]

mesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='polygon')

# define vemspace
p = 1
vemspace =VirtualElementSpace2d(mesh, p)

# get_sys_parameter

#option.maxit = 5000
option.fA = 0.5
option.chiN = 15
option.Nh= n
chiN = option.Ndeg*option.chiAB

# define fields
Ndof = vemspace.number_of_global_dofs()
fields = np.zeros((Ndof, 2))
node = mesh.node


if fieldsType == 1:
    fields[:, 0] = chiN*(-1 + 2*np.random.rand(1, Ndof))
    fields[:, 1] = chiN*(-1 + 2*np.random.rand(1, Ndof))
elif fieldsType == 2:
    pi = np.pi
    fields[:, 1] = chiN * (np.sin(3*node[:,0]) + np.cos(3*node[:, 1]))
elif fieldsType == 3:
    pi = np.pi
    fields[:, 1] = chiN * (np.sin(3 + np.cos(pi/5))
elif fieldsType == 4:
    fields[:: 25, 1] = 1 
option.fields = fields

scft = SCFTVEMModel(vemspace, option) 


scft.initialize()
scft.find_saddle_point(datafile='datafile', file_path='./results/')

