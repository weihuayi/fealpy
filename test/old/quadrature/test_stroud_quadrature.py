import sys
import time
import numpy as np
# scipy 的积分函数
import scipy.integrate as integrate
from fealpy.mesh import TriangleMesh, TetrahedronMesh, IntervalMesh
from fealpy.quadrature import StroudQuadrature
from fealpy.functionspace import BernsteinFESpace

def f3(x, y, z):
    return np.cos(10*x)*np.sin(10*y)*np.exp(10*z)*x**7*y**2*z**3

def f(x, y):
    return np.cos(10*x)*np.sin(10*y)*x**10*y**2

def f1(x):
    return np.cos(10*x)*x**12

N = int(sys.argv[1])

tetmesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], N, N, N)
qf = StroudQuadrature(3, 7)
bcs, weights = qf.get_points_and_weights()
#bcs, weights = tetmesh.integrator(7).get_quadrature_points_and_weights() 
print(bcs.shape[0])
points = tetmesh.bc_to_point(bcs)
val = f3(points[..., 0], points[..., 1], points[..., 2]) #(NQ, NC)

space = BernsteinFESpace(tetmesh, 10)
phi = space.basis(bcs) #(NQ, NC, ldof)

cm = tetmesh.entity_measure('cell')
I = np.einsum('qc, q, c->', val, weights, cm, optimize=True)
I0 = integrate.tplquad(f3, 0, 1, 0, 1, 0, 1)[0]
print(I)
print(I0)
print(abs(I-I0))
print('-----------------------------------')

#t = time.time()
#NC = tetmesh.number_of_cells()
#N = NC//10
#for i in range(9):
#    points = tetmesh.bc_to_point(bcs)[:, i*N:(i+1)*N]
#    val = f3(points[..., 0], points[..., 1], points[..., 2]) #(NQ, NC)
#    phi0 = phi[:, i*N:(i+1)*N]
#    cm0 = cm[i*N:(i+1)*N]
#    vvv = np.einsum('qcd, qc, q, c->d', phi0, val, weights, cm0, optimize=True)

#vvv = np.einsum('qcd, qc, q, c->d', phi, val, weights, cm, optimize=True)
s = time.time() - t
print("time:", s)



tmesh = TriangleMesh.from_box([0, 1, 0, 1], N, N)
qf = StroudQuadrature(2, 7)
bcs, weights = qf.get_points_and_weights()
points = tmesh.bc_to_point(bcs)
val = f(points[..., 0], points[..., 1]) #(NQ, NC)

cm = tmesh.entity_measure('cell')
I = np.einsum('qc, q, c->', val, weights, cm)
I0 = integrate.dblquad(f, 0, 1, 0, 1)[0] 
print(I)
print(I0)
print(abs(I-I0))
print('-----------------------------------')


imesh = IntervalMesh.from_interval([0, 1], N)
qf = StroudQuadrature(1, 7)
bcs, weights = qf.get_points_and_weights()
points = imesh.bc_to_point(bcs)
val = f1(points[..., 0]) #(NQ, NC)
cm = imesh.entity_measure('cell')
I = np.einsum('qc, q, c->', val, weights, cm)
I0 = integrate.quad(f1, 0, 1)[0]
print(I)
print(I0)
print(abs(I-I0))
print('-----------------------------------')

















