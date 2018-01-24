import numpy as np
import types

class FunctionNorm():
    def __init__(self, integrator, area):
        self.integrator = integrator
        self.area = area

    def L2_norm(self, u, mesh, barycenter=False, elemtype=False):
        qf = self.integrator  
        bcs, ws = qf.quadpts, qf.weights
        if barycenter is False:
            pp = mesh.bc_to_point(bcs)
            val = u(pp)
        else:
            val = u(bcs)

        e = val**2
        if len(e.shape) == 2:
            e = np.sum(e, axis=1)
        e *= self.area 
        if elemtype is True:
            return np.sqrt(e)
        else:
            return np.sqrt(e.sum()) 

    def L2_error(self, u, uh, mesh=None, barycenter=True):
        if mesh is None:
            mesh = uh.V.mesh

        qf = self.integrator  
        bcs, ws = qf.quadpts, qf.weights
        pp = mesh.bc_to_point(bcs)

        if barycenter is True:
            val0 = uh(bcs)
        else:
            val0 = uh(pp)
        val1 = u(pp)
        
        e = (val1 - val0)**2
        axis = tuple(range(2, len(e.shape)))
        e = np.sum(e, axis=axis)
        e = np.einsum('i, ij->j', ws, e)
        e *= self.area 
        return np.sqrt(e.sum()) 

    def l2_error(self, u, uh):
        uI = uh.V.interpolation(u)
        gdof = uh.V.number_of_global_dofs()
        return np.sqrt(np.sum((uI - uh)**2)/gdof)

    def infty_error(self, u, uh):
        uI = uh.V.interpolation(u)
        gdof = uh.V.number_of_global_dofs()
        return np.max(np.abs(uI - uh))

