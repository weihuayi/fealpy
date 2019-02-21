import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.io as sio

from fealpy.pde.BiharmonicModel2d import VSSPData 
from fealpy.fem.BiharmonicFEMModel import BiharmonicRecoveryFEMModel
from fealpy.fem.doperator import mass_matrix
from scipy.sparse.linalg import eigs, eigsh, LinearOperator

import pyamg

maxit = 8 
pde = VSSPData()

mesh = pde.init_mesh(meshtype='tri', n=2)
fname = 'datacp'

for i in range(maxit):
    fem = BiharmonicRecoveryFEMModel(mesh, pde, 1, 5, rtype='simple')
    # A = fem.get_laplace_matrix()
    A = fem.get_laplace_matrix() + fem.get_neuman_penalty_matrix()
    M = fem.space.mass_matrix(fem.integrator, fem.measure)
    isBdDof = fem.space.boundary_dof()

    A = A[~isBdDof, :][:, ~isBdDof]
    M = M[~isBdDof, :][:, ~isBdDof]
    A = A.tocsr()
    M = M.tocsr()

    mesh = fem.space.mesh
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    data = {"A":A, "M":M, 'node':node, 'elem':cell+1, 'flag':isBdDof};
    sio.matlab.savemat(fname+str(i)+'.mat', data)

    if i < maxit-1:
        mesh.uniform_refine()
    continue

    if True:
        ml = pyamg.ruge_stuben_solver(M)
        def matvec(b):
            return ml.solve(b, tol=1e-12, accel='cg')
        op = LinearOperator(A.shape, matvec=matvec)
        eigens, eivector = eigsh(A, k=3,  M=M, which='SM', tol=1e-12, Minv=op)
        print(eigens)
    else:
        ml = pyamg.ruge_stuben_solver(A)
        def matvec(b):
            return ml.solve(M@b, tol=1e-12, accel='cg')
        op = LinearOperator(A.shape, matvec=matvec)
        eigens, eivector = eigsh(op, k=3, tol=1e-12)
        print(np.sort(1/eigens))



fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
axes.set_axis_on()
plt.show()

