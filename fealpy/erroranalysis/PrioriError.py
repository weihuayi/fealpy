import numpy as np
from ..quadrature import TriangleQuadrature


def L2_norm(u, V, order=4, dtype=np.float):
    mesh = V.mesh
    NC = mesh.number_of_cells()
    qf = TriangleQuadrature(order)
    nQuad = qf.get_number_of_quad_points()

    gdof = V.number_of_global_dofs()
    ldof = V.number_of_local_dofs()
    
    e = np.zeros((NC,), dtype=dtype)
    for i in range(nQuad):
        lambda_k, w_k = qf.get_gauss_point_and_weight(i)
        p = mesh.bc_to_point(lambda_k)
        uval = u(p)
        if len(uval.shape) == 1:
            e += w_k*uval**2
        else:
            e += w_k*(uval**2).sum(axis=1)
    e *= mesh.area()
    return np.sqrt(e.sum()) 

def l2_error(u, uh):
    uI = V.interpolation(u)
    gdof = V.number_of_global_dofs()
    return np.sqrt(np.sum((uI - uh)**2)/gdof)

def L2_error(u, uh, order=4, dtype=np.float):
    V = uh.V
    mesh = V.mesh

    NC = mesh.number_of_cells()
    qf = TriangleQuadrature(order)
    nQuad = qf.get_number_of_quad_points()

    gdof = V.number_of_global_dofs()
    ldof = V.number_of_local_dofs()
    
    e = np.zeros((NC,), dtype=dtype)
    for i in range(nQuad):
        lambda_k, w_k = qf.get_gauss_point_and_weight(i)
        uhval = uh.value(lambda_k)
        p = mesh.bc_to_point(lambda_k)
        uval = u(p)
        if len(uval.shape) == 1:
            e += w_k*(uhval - uval)*(uhval - uval)
        else:
            e += w_k*((uhval - uval)*(uhval - uval)).sum(axis=1)
    e *= mesh.area()
    #isInCell = ~mesh.ds.boundary_cell_flag()
    #return np.sqrt(e[isInCell].sum())
    return np.sqrt(e.sum()) 

def div_error(f, ruh, order=4, dtype=np.float):
    V = ruh.V
    mesh = V.mesh

    NC = mesh.number_of_cells()
    qf = TriangleQuadrature(order)
    nQuad = qf.get_number_of_quad_points()

    gdof = V.number_of_global_dofs()
    ldof = V.number_of_local_dofs()
    
    e = np.zeros((NC,), dtype=dtype)
    for i in range(nQuad):
        lambda_k, w_k = qf.get_gauss_point_and_weight(i)
        uhval = ruh.div_value(lambda_k)
        p = mesh.bc_to_point(lambda_k)
        uval = f(p)
        e += w_k*(uhval - uval)*(uhval - uval)
    e *= mesh.area()
    return np.sqrt(e.sum())

def H1_semi_error(gu, uh, order=3, dtype=np.float):
    V = uh.V
    mesh = V.mesh

    NC = mesh.number_of_cells()
    qf = TriangleQuadrature(order)
    nQuad = qf.get_number_of_quad_points()

    gdof = V.number_of_global_dofs()
    ldof = V.number_of_local_dofs()
    
    e = np.zeros((NC,), dtype=dtype)
    for i in range(nQuad):
        lambda_k, w_k = qf.get_gauss_point_and_weight(i)
        gval = uh.grad_value(lambda_k)
        p = mesh.bc_to_point(lambda_k)
        val = gu(p)
        e += w_k*((gval - val)*(gval - val)).sum(axis=1)
    e *= mesh.area()
    return np.sqrt(e.sum())

def H1_error(u, uh):
    pass

