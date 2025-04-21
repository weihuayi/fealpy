
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace.huzhang_fe_space import HuZhangFESpace 
from fealpy.functionspace import LagrangeFESpace

from scipy.sparse import csr_matrix, coo_matrix, bmat
from scipy.sparse.linalg import spsolve

from linear_elastic_pde import LinearElasticPDE

from sympy import symbols, sin, cos, Matrix, lambdify

from fealpy.tools.show import showmultirate
from fealpy.tools.show import show_error_table

from scipy.sparse import csr_matrix
from mumps import DMumpsContext
from scipy.sparse.linalg import minres, gmres, lgmres

import sys
import time

def Solve(A, b):
    ctx = DMumpsContext()
    ctx.set_silent()
    ctx.set_centralized_sparse(A)

    x = bm.array(b)

    ctx.set_rhs(x)
    ctx.run(job=6)
    ctx.destroy()

    x = list(x)
    #x, _ = lgmres(A, b)
    return x


def plot_linear_function(uh, u):
    fig = plt.figure()
    fig.set_facecolor('white')
    axes = plt.axes(projection='3d')

    NC = mesh.number_of_cells()

    mid = mesh.entity_barycenter("cell")
    node = mesh.entity("node")
    cell = mesh.entity("cell")

    coor = node[cell]
    val = u(node).reshape(-1) 

    bcs = bm.eye(3, dtype=uh.dtype)
    uhval = uh(bcs)[..., 2] # (NC, 3)
    for ii in range(NC):
        axes.plot_trisurf(coor[ii, :, 0], coor[ii, :, 1], uhval[ii], color = 'r', lw=0.0)#数值解图像

    fig = plt.figure()
    fig.set_facecolor('white')
    axes = plt.axes(projection='3d')
    for ii in range(NC):
        axes.plot_trisurf(coor[ii, :, 0], coor[ii, :, 1], val[cell[ii]], color = 'b', lw=0.0)
    plt.show()

def mass_matrix(space : HuZhangFESpace, lambda_0 : float, mu_0 : float):
    p = space.p
    mesh = space.mesh
    gdof = space.number_of_global_dofs()

    cellmeasure = mesh.entity_measure('cell')
    qf = mesh.quadrature_formula(p+2, 'cell')

    bcs, ws = qf.get_quadrature_points_and_weights()
    phi = space.basis(bcs)
    trphi = phi[..., 0] + phi[..., -1]

    num = bm.array([1, 2, 1], dtype=phi.dtype)
    A = lambda0*bm.einsum('q, c, cqld, cqmd, d->clm', ws, cellmeasure, phi, phi, num)
    A -= lambda1*bm.einsum('q, c, cql, cqm->clm', ws, cellmeasure, trphi, trphi)

    cell2dof = space.cell_to_dof()
    I = bm.broadcast_to(cell2dof[:, None], A.shape)
    J = bm.broadcast_to(cell2dof[..., None], A.shape)
    A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof),
                   dtype=phi.dtype)
    return A

def mix_matrix(space0 : HuZhangFESpace, space1 : LagrangeFESpace):
    p = space0.p
    mesh = space0.mesh
    gdof0 = space0.number_of_global_dofs()
    gdof1 = space1.number_of_global_dofs()

    cell2dof0 = space0.cell_to_dof()
    cell2dof1 = space1.cell_to_dof()

    cellmeasure = mesh.entity_measure('cell')
    qf = mesh.quadrature_formula(p+2, 'cell')

    bcs, ws = qf.get_quadrature_points_and_weights()
    phi = space0.div_basis(bcs)
    psi = space1.basis(bcs)
    B_ = bm.einsum('q, c, cqld, cqm->clmd', ws, cellmeasure, phi, psi)

    shape = B_.shape[:-1]

    B = coo_matrix((gdof0, gdof1*2), dtype=phi.dtype)
    I = bm.broadcast_to(cell2dof0[..., None], shape)
    for i in range(2):
        J = bm.broadcast_to(gdof1*i + cell2dof1[:, None], shape)
        B += coo_matrix((B_[..., i].flat, (I.flat, J.flat)), shape=(gdof0, gdof1*2), dtype=phi.dtype)
    return B.tocsr()

def source_vector(space : LagrangeFESpace, f : callable):
    p = space.p
    mesh = space.mesh
    gdof = space.number_of_global_dofs()

    cellmeasure = mesh.entity_measure('cell')
    qf = mesh.quadrature_formula(p+2, 'cell')

    bcs, ws = qf.get_quadrature_points_and_weights()
    points = mesh.bc_to_point(bcs)

    phi  = space.basis(bcs)
    fval = f(points)
    b = bm.einsum('q, c, cql, cqd->cld', ws, cellmeasure, phi, fval)

    cell2dof = space.cell_to_dof()
    r = bm.zeros(gdof*2, dtype=phi.dtype)
    for i in range(2):
        bm.add.at(r, gdof*i + cell2dof, b[..., i]) 
    return r

def displacement_boundary_condition(space : HuZhangFESpace, g : callable):
    p = space.p
    mesh = space.mesh
    TD = mesh.top_dimension()
    ldof = space.dof.number_of_local_dofs()
    gdof = space.dof.number_of_global_dofs()

    bdedge = mesh.boundary_edge_flag()
    e2c = mesh.edge_to_cell()[bdedge]
    en  = mesh.edge_unit_normal()[bdedge]
    cell2dof = space.cell_to_dof()[e2c[:, 0]]
    NBF = bdedge.sum()

    cellmeasure = mesh.entity_measure('edge')[bdedge]
    qf = mesh.quadrature_formula(p+2, 'edge')

    bcs, ws = qf.get_quadrature_points_and_weights()
    NQ = len(bcs)

    bcsi = [bm.insert(bcs, i, 0, axis=-1) for i in range(3)]

    symidx = [[0, 1], [1, 2]]
    phin = bm.zeros((NBF, NQ, ldof, 2), dtype=space.ftype)
    gval = bm.zeros((NBF, NQ, 2), dtype=space.ftype)
    for i in range(3):
        flag = e2c[:, 2] == i
        phi = space.basis(bcsi[i])[e2c[flag, 0]] 
        phin[flag, ..., 0] = bm.sum(phi[..., symidx[0]] * en[flag, None, None], axis=-1)
        phin[flag, ..., 1] = bm.sum(phi[..., symidx[1]] * en[flag, None, None], axis=-1)
        points = mesh.bc_to_point(bcsi[i])[e2c[flag, 0]]
        gval[flag] = g(points)

    b = bm.einsum('q, c, cqld, cqd->cl', ws, cellmeasure, phin, gval)
    cell2dof = space.cell_to_dof()[e2c[:, 0]]
    r = bm.zeros(gdof, dtype=phi.dtype)
    bm.add.at(r, cell2dof, b) 
    return r

def solve(pde, N, p):
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=N, ny=N)
    space0 = HuZhangFESpace(mesh, p=p)
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

    X = spsolve(A, F)

    sigmaval = X[:gdof0]
    u0val = X[gdof0:gdof0+gdof1]
    u1val = X[gdof0+gdof1:]

    sigmah = space0.function()
    sigmah[:] = sigmaval

    uh0 = space1.function()
    uh1 = space1.function()
    uh0[:] = u0val
    uh1[:] = u1val

    return sigmah, uh0, uh1


if __name__ == "__main__":
    lambda0 = 4
    lambda1 = 1
    maxit = 5
    p = int(sys.argv[1])

    errorType = [
                 '$|| \\boldsymbol{\\sigma} - \\boldsymbol{\\sigma}_h||_{\\Omega,0}$',
                 '$|| \\boldsymbol{u} - \\boldsymbol{u}_h||_{\\Omega,0}$',
                 ]
    errorMatrix = bm.zeros((2, maxit), dtype=bm.float64)
    h = bm.zeros(maxit, dtype=bm.float64)

    x, y = symbols('x y')


    pi = bm.pi 
    u0 = (sin(pi*x)*sin(pi*y))**2
    u1 = (sin(pi*x)*sin(pi*y))**2
    u0 = sin(5*x)*sin(7*y)
    u1 = cos(5*x)*cos(4*y)

    u = [u0, u1]
    pde = LinearElasticPDE(u, lambda0, lambda1)

    for i in range(maxit):
        N = 2**(i+1) 
        sigmah, uh0, uh1 = solve(pde, N, p)
        mesh = sigmah.space.mesh

        u0 = lambda p : pde.displacement(p)[..., 0]
        u1 = lambda p : pde.displacement(p)[..., 1]

        e0 = mesh.error(uh0, u0)
        e1 = mesh.error(uh1, u1)
        e2 = mesh.error(sigmah, pde.stress)

        h[i] = 1/N
        errorMatrix[0, i] = e2
        errorMatrix[1, i] = bm.sqrt(e0**2 + e1**2)
        print(N, e0, e1, e2)

        #stress0 = lambda p : pde.stress(p)[..., 0]
        #stress1 = lambda p : pde.stress(p)[..., 1]
        #stress2 = lambda p : pde.stress(p)[..., 2]

        #plot_linear_function(sigmah, stress2)

    show_error_table(h, errorType, errorMatrix)
    showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
    plt.show()























