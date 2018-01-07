import numpy as np
from ..quadrature import TriangleQuadrature

class PrioriError():
    def __init__(self):
        pass

    def L2_error(self, u, uh, order=4):
        mesh = uh.V.mesh

        NC = mesh.number_of_cells()
        qf = TriangleQuadrature(order)
        bcs, ws = qf.quadpts, qf.weights
        uhval = uh.value(bcs)
        pp = mesh.bc_to_point(bcs)
        uval = u(pp)
        e = np.einsum('i, ij->j', ws, (uhval - uval)**2)
        e *=mesh.area()
        return np.sqrt(e.sum()) 

    def l2_error(self, u, uh):
        uI = V.interpolation(u)
        gdof = V.number_of_global_dofs()
        return np.sqrt(np.sum((uI - uh)**2)/gdof)

    def H1_semi_error(self, gu, uh, order=3, gradfunction=False):
        mesh = uh.V.mesh

        qf = TriangleQuadrature(order)
        bcs, ws = qf.quadpts, qf.weights
        if gradfunction is True:
            guh = uh.value(bcs)
        else:
            guh = uh.grad_value(bcs)
        pp = mesh.bc_to_point(bcs)
        gu = gu(pp)
        e = np.sum((guh - gu)**2, axis=-1)
        e = np.einsum('i, ij->j', ws, e)
        e *=mesh.area()
        return np.sqrt(e.sum()) 

    def H1_error(self, u, gu, uh, order=3):
        e0 = self.L2_error(u, uh, order=order)
        e1 = self.H1_semi_error(gu, uh, order=order)
        return np.sqrt(e0**2 + e1**2)

    def div_error(self, du, rguh, order=4, dtype=np.float):
        mesh = uh.V.mesh
        qf = TriangleQuadrature(order)
        bcs, ws = qf.quadpts, qf.weights
        val0 = guh.div_value(bcs)
        pp = mesh.bc_to_point(bcs)
        val1 = du(pp)
        e = np.einsum('i, ij->j', ws, (val1 - val0)**2)
        e *=mesh.area()
        return np.sqrt(e.sum()) 

        

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





