import sys
import numpy as np

from fealpy.functionspace.vem_space import VirtualElementSpace2d 
from SCFTVEMModel import SCFTVEMModel, SCFTParameter
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh 


n = int(sys.argv[1])  # n is mesh refine
model_pmt = int(sys.argv[2]) 
fieldsType = int(sys.argv[3]) # define fields type

# define squre mesh size
t = 6*np.pi
box = [0, t, 0, t]
mesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='polygon')

# define vemspace
p = 1
vemspace =VirtualElementSpace2d(mesh, p)

# get_sys_parameter
option = SCFTParameter()
option.maxit = 5000

if model_pmt == 1:
    option.fA = 0.2
    option.chiN = 25
elif model_pmt == 2:
    option.fA = 0.5
    option.chiN = 15

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
    fields[:, 1] = chiN*(np.sin(np.cos(pi/6)*node[:,1]) + np.cos(np.sin(pi/6)*node[:,1])) 
   
option.fields = fields

scft = SCFTVEMModel(vemspace, option) 
scft.initialize()
scft.find_saddle_point()

