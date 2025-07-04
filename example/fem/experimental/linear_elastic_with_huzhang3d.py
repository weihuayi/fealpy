
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TetrahedronMesh
from huzhang_fe_space3d import HuZhangFESpace3d
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace.functional import symmetry_span_array, symmetry_index

from scipy.sparse import csr_matrix, coo_matrix, bmat
from scipy.sparse.linalg import spsolve

from linear_elastic_pde import LinearElasticPDE3d

from sympy import symbols, sin, cos, Matrix, lambdify

from fealpy.tools.show import showmultirate
from fealpy.tools.show import show_error_table

import sys
import time

from scipy.sparse import csr_matrix
from mumps import DMumpsContext
from scipy.sparse.linalg import minres, gmres, lgmres

def Solve(A, b):
    #ctx = DMumpsContext()
    #ctx.set_silent()
    #ctx.set_centralized_sparse(A)

    #x = bm.array(b)

    #ctx.set_rhs(x)
    #ctx.run(job=6)
    #ctx.destroy()

    #x = list(x)
    x, _ = lgmres(A, b)
    return x

def mass_matrix(space : HuZhangFESpace3d, lambda_0 : float, mu_0 : float):
    p = space.p
    mesh = space.mesh
    TD = mesh.top_dimension()
    gdof = space.number_of_global_dofs()

    cellmeasure = mesh.entity_measure('cell')
    qf = mesh.quadrature_formula(p+3, 'cell')

    bcs, ws = qf.get_quadrature_points_and_weights()
    phi = space.basis(bcs)
    trphi = phi[..., 0] + phi[..., 3] + phi[..., -1]

    _, num = symmetry_index(d=TD, r=2)
    A = lambda0*bm.einsum('q, c, cqld, cqmd, d->clm', ws, cellmeasure, phi, phi, num)
    A -= lambda1*bm.einsum('q, c, cql, cqm->clm', ws, cellmeasure, trphi, trphi)

    cell2dof = space.cell_to_dof()
    I = bm.broadcast_to(cell2dof[:, None], A.shape)
    J = bm.broadcast_to(cell2dof[..., None], A.shape)
    A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof),
                   dtype=phi.dtype)
    return A

def mix_matrix(space0 : HuZhangFESpace3d, space1 : LagrangeFESpace):
    p = space0.p
    mesh = space0.mesh
    TD = mesh.top_dimension()
    gdof0 = space0.number_of_global_dofs()
    gdof1 = space1.number_of_global_dofs()

    cell2dof0 = space0.cell_to_dof()
    cell2dof1 = space1.cell_to_dof()

    cellmeasure = mesh.entity_measure('cell')
    qf = mesh.quadrature_formula(p+3, 'cell')

    bcs, ws = qf.get_quadrature_points_and_weights()
    phi = space0.div_basis(bcs)
    psi = space1.basis(bcs)
    B_ = bm.einsum('q, c, cqld, cqm->clmd', ws, cellmeasure, phi, psi)

    shape = B_.shape[:-1]

    B = coo_matrix((gdof0, gdof1*TD), dtype=phi.dtype)
    I = bm.broadcast_to(cell2dof0[..., None], shape)
    for i in range(TD):
        J = bm.broadcast_to(gdof1*i + cell2dof1[:, None], shape)
        B += coo_matrix((B_[..., i].flat, (I.flat, J.flat)), shape=(gdof0, gdof1*TD), dtype=phi.dtype)
    return B.tocsr()

def source_vector(space : LagrangeFESpace, f : callable):
    p = space.p
    mesh = space.mesh
    TD = mesh.top_dimension()
    gdof = space.number_of_global_dofs()

    cellmeasure = mesh.entity_measure('cell')
    qf = mesh.quadrature_formula(p+3, 'cell')

    bcs, ws = qf.get_quadrature_points_and_weights()
    points = mesh.bc_to_point(bcs)

    phi  = space.basis(bcs)
    fval = f(points)
    b = bm.einsum('q, c, cql, cqd->cld', ws, cellmeasure, phi, fval)

    cell2dof = space.cell_to_dof()
    r = bm.zeros(gdof*TD, dtype=phi.dtype)
    for i in range(TD):
        bm.add.at(r, gdof*i + cell2dof, b[..., i]) 
    return r

def displacement_boundary_condition(space : HuZhangFESpace3d, g : callable):
    p = space.p
    mesh = space.mesh
    TD = mesh.top_dimension()
    ldof = space.number_of_local_dofs()
    gdof = space.number_of_global_dofs()

    bdface = mesh.boundary_face_index()
    f2c = mesh.face_to_cell()[bdface]
    fn  = mesh.face_unit_normal()[bdface]
    cell2dof = space.cell_to_dof()[f2c[:, 0]]
    NBF = len(bdface)

    cellmeasure = mesh.entity_measure('face')[bdface]
    qf = mesh.quadrature_formula(p+3, 'face')

    bcs, ws = qf.get_quadrature_points_and_weights()
    NQ = len(bcs)

    bcsi = [bm.insert(bcs, i, 0, axis=-1) for i in range(4)]

    symidx = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
    phin = bm.zeros((NBF, NQ, ldof, 3), dtype=space.ftype)
    gval = bm.zeros((NBF, NQ, 3), dtype=space.ftype)
    for i in range(4):
        flag = f2c[:, 2] == i
        phi = space.basis(bcsi[i])[f2c[flag, 0]] 
        phin[flag, ..., 0] = bm.sum(phi[..., symidx[0]] * fn[flag, None, None], axis=-1)
        phin[flag, ..., 1] = bm.sum(phi[..., symidx[1]] * fn[flag, None, None], axis=-1)
        phin[flag, ..., 2] = bm.sum(phi[..., symidx[2]] * fn[flag, None, None], axis=-1)
        points = mesh.bc_to_point(bcsi[i])[f2c[flag, 0]]
        gval[flag] = g(points)

    b = bm.einsum('q, c, cqld, cqd->cl', ws, cellmeasure, phin, gval)
    cell2dof = space.cell_to_dof()[f2c[:, 0]]
    r = bm.zeros(gdof, dtype=phi.dtype)
    bm.add.at(r, cell2dof, b) 
    return r

def solve(pde, N, p=4):
    mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=N, ny=N, nz=N)
    space0 = HuZhangFESpace3d(mesh, p=p)
    space1 = LagrangeFESpace(mesh, p=p-1, ctype='D')

    lambda0 = pde.lambda0
    lambda1 = pde.lambda1

    gdof0 = space0.number_of_global_dofs()
    gdof1 = space1.number_of_global_dofs()

    A = mass_matrix(space0, lambda0, lambda1)
    B = mix_matrix(space0, space1)
    b = source_vector(space1, pde.source)
    a = displacement_boundary_condition(space0, pde.displacement)

    A = bmat([[A, B], [B.T, None]], format='csr', dtype=A.dtype) 

    F = bm.zeros(A.shape[0], dtype=A.dtype)
    F[:gdof0] = a
    F[gdof0:] = -b

    print('begin to solve linear system')
    X =Solve(A, F)
    #X = spsolve(A, F)
    print('end to solve linear system')

    sigmaval = X[:gdof0]
    u0val = X[gdof0:gdof0+gdof1]
    u1val = X[gdof0+gdof1:gdof0+2*gdof1]
    u2val = X[gdof0+2*gdof1:]

    sigmah = space0.function()
    sigmah[:] = sigmaval

    uh0 = space1.function()
    uh1 = space1.function()
    uh2 = space1.function()
    uh0[:] = u0val
    uh1[:] = u1val
    uh2[:] = u2val

    return sigmah, uh0, uh1, uh2


if __name__ == "__main__":
    lambda0 = 4
    lambda1 = 1
    maxit = 4

    errorType = [
                 '$|| \\boldsymbol{\\sigma} - \\boldsymbol{\\sigma}_h||_{\\Omega,0}$',
                 '$|| \\boldsymbol{u} - \\boldsymbol{u}_h||_{\\Omega,0}$',
                 ]
    errorMatrix = bm.zeros((2, maxit), dtype=bm.float64)
    h = bm.zeros(maxit, dtype=bm.float64)

    for i in range(maxit):
        N = 2**(i+0) 

        x, y, z = symbols('x y z')


        pi = bm.pi 
        u0 = sin(pi*x)*sin(pi*y)*sin(pi*z)
        u1 = sin(pi*x)*sin(pi*y)*sin(pi*z)
        u2 = sin(pi*x)*sin(pi*y)*sin(pi*z)

        #u0 = x**4
        #u1 = y**4
        #u2 = z**4

        u = [u0, u1, u2]
        pde = LinearElasticPDE3d(u, lambda0, lambda1)

        sigmah, uh0, uh1, uh2 = solve(pde, N)
        mesh = sigmah.space.mesh

        u0 = lambda p : pde.displacement(p)[..., 0]
        u1 = lambda p : pde.displacement(p)[..., 1]
        u2 = lambda p : pde.displacement(p)[..., 2]

        e0 = mesh.error(uh0, u0)
        e1 = mesh.error(uh1, u1)
        e2 = mesh.error(uh2, u2)
        e3 = mesh.error(sigmah, pde.stress)

        h[i] = 1/N
        errorMatrix[0, i] = e3
        errorMatrix[1, i] = bm.sqrt(e0**2 + e1**2 + e2**2)
        print('error:', errorMatrix[:, i])

        bcs = bm.eye(4)
        sigmahval = sigmah.value(bcs)
        #print(sigmahval)
        #print(pde.stress(mesh.bc_to_point(bcs)))

        #stress0 = lambda p : pde.stress(p)[..., 0]
        #stress1 = lambda p : pde.stress(p)[..., 1]
        #stress2 = lambda p : pde.stress(p)[..., 2]

        #plot_linear_function(sigmah, stress2)

    show_error_table(h, errorType, errorMatrix)
    #showmultirate(plt, 1, h, errorMatrix,  errorType, propsize=20)
    #plt.show()























