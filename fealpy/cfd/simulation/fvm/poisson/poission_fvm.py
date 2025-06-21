import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.uniform_mesh_2d import UniformMesh2d
from fealpy.mesh.quadrangle_mesh import QuadrangleMesh
from scipy.sparse import coo_matrix
from fealpy.pde.poisson_2d import CosCosData
from scipy.sparse.linalg import spsolve


def solution(p):
    x = p[..., 0]
    y = p[..., 1]
    pi = np.pi
    return np.cos(pi*x)*np.cos(pi*y)
def source(p):
    x = p[..., 0]
    y = p[..., 1]
    pi = np.pi
    return 2*pi**2*np.cos(pi*x)*np.cos(pi*y) 
def dirichlet(p):
    return solution(p)
def is_dirichlet_boundary(p):
    eps = 1e-12
    x = p[..., 0]
    y = p[..., 1]
    return (np.abs(y-1)<eps)|(np.abs(x-1)<eps)|(np.abs(x)<eps)|(np.abs(y)<eps)

nx = 80
ny = 80
domain = [0, 1, 0, 1]
mesh = QuadrangleMesh.from_box(box=[0,1,0,1],nx=nx,ny=ny)
NC = mesh.number_of_cells()
node = mesh.entity('node')
edge = mesh.entity('edge')
cell2edge = mesh.cell_to_edge()



h = 1/nx

I = np.arange(NC)
J = np.arange(NC)
val = 4*h/h*np.ones(NC)
A0 = coo_matrix((val,(I, J)), shape=(NC, NC))


e2e = mesh.cell_to_cell()
flag = e2e[np.arange(NC)] == np.arange(NC)[:, None] # 判断是边界的

I = np.where(flag)[0] 
J = np.where(flag)[0]
val = (h/(h/2)-h/h)*np.ones(I.shape)
A0 += coo_matrix((val,(I, J)), shape=(NC, NC))


I = np.where(~flag)[0] 
J = e2e[~flag]
val = -h/h*np.ones(I.shape)
A0 += coo_matrix((val,(I, J)), shape=(NC, NC))
print(A0.toarray())


b = np.zeros(NC)
index = np.where(flag)[0]
index1 = cell2edge[flag]
point = node[edge[index1]]
point = point[:,0,:]*(1/2)+point[:,1,:]*(1/2)
flag = is_dirichlet_boundary(point)
bu = dirichlet(point)[flag]
data = bu*h/(h/2)
np.add.at(b, index, data)


bb = mesh.integral(source, celltype=True)
print('bb',bb)
print('bb+b',bb+b)
uh = spsolve(A0, bb+b)
print('uh',uh)

ipoint = mesh.entity_barycenter('cell')
u = solution(ipoint)
e = u - uh
print('emax', np.max(np.abs(u-uh)))
print('eL2', np.sqrt(np.sum(h*h*e**2)))


fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')
ipoint = mesh.entity_barycenter('cell')
xx = ipoint[..., 0]
yy = ipoint[..., 1]
X = xx.reshape(nx, ny)
Y = yy.reshape(ny, ny)
Z = uh.reshape(nx, ny)
ax1.plot_surface(X, Y, Z, cmap='rainbow')
plt.show()
