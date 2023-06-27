import numpy as np
import matplotlib.pyplot as plt
from fealpy.pde.darcy_forchheimer_2d import DarcyForchheimerdata1
from fealpy.functionspace.lagrange_fem_space import VectorLagrangeFiniteElementSpace

box = [0, 1, 0, 1]
mu = 1
rho = 1
beta = 10
alpha = 1/beta
tol = 1e-6
level = 1
mg_maxN = 1
maxN = 2000
p = 1

pde = DarcyForchheimerdata1(box,mu,rho,beta,alpha,level,tol,maxN,mg_maxN)
mesh = pde.init_mesh(level)
node = mesh.node
cell = mesh.ds.cell
print('node',node)
print('node0',node[cell[:,0]])
print('node1',node[cell[:,1]])
print('node2',node[cell[:,2]])
NC = mesh.number_of_cells()
NN = mesh.number_of_nodes()
cellmeasure = mesh.entity_measure('cell')

integrator = mesh.integrator(p+2)
bcs = integrator.quadpts
ws = integrator.weights
pxy = np.zeros((6,NC,2),dtype=np.float)
for i in range(6):
    pxy[i,:,:] = bcs[i,0]*node[cell[:,0],:] + bcs[i,1]*node[cell[:,1]]\
            + bcs[i,2]*node[cell[:,2]]
V = VectorLagrangeFiniteElementSpace(mesh, p, spacetype='D')
b = V.source_vector(pde.f, integrator, cellmeasure)
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
plt.show()
