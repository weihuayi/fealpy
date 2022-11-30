import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh2d
from fealpy.mesh import MeshFactory as MF 
from fealpy.functionspace import LagrangeFiniteElementSpace

def circle(p):
    x = p[...,0]
    y = p[...,1]
    val = np.sqrt((x-0.5)**2+(y-0.5)**2)-1
    return val

ns = 2
domain= [0, 1, 0, 1]
nx = 2*ns
ny = ns

extent = [0, 2*nx, 0, 2*ny]
origin = (domain[0], domain[2])
hx = (domain[1] - domain[0])/(extent[1] - extent[0])
hy = (domain[3] - domain[2])/(extent[3] - extent[2])
qmesh = UniformMesh2d(extent, h=(hx, hy), origin=origin) #四边形网格

tmesh = MF.boxmesh2d(domain, nx=nx, ny=ny) #三角性网格   

uspace = LagrangeFiniteElementSpace(tmesh, p=2)   


phi0 = qmesh.function('node') 

ips = uspace.interpolation_points()
phi1 = circle(ips)

NN = tmesh.number_of_nodes()
NE = tmesh.number_of_edges()

phi0[0::2, 0::2].flat[:] = phi1[0:NN]




fig = plt.figure()
axes = fig.gca()

qmesh.add_plot(axes)
qmesh.find_node(axes, node=node, showindex=True)

fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes)
tmesh.find_node(axes, node=ips, showindex=True)
plt.show()

