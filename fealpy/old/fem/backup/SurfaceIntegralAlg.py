import numpy as np

class SurfaceIntegralAlg():
    def __init__(self, integrator, mesh, area):
        self.integrator = integrator
        self.mesh = mesh
        self.area = area

    def integral(self, u, celltype=False, barycenter=True):
        qf = self.integrator  
        bcs, ws = qf.quadpts, qf.weights
        if barycenter:
            val = u(bcs)
        else:
            pp = self.mesh.bc_to_point(bcs)
            val = u(pp)
        e = np.einsum('i, ij..., j->j...', ws, val, self.area)
        if celltype is True:
            return e
        else:
            return e.sum() 

    def L2_error(self, u, uh, celltype=False):
        def f(x):
            xx = self.mesh.bc_to_point(x)
            xx, _ = self.mesh.surface.project(xx)
            return (u(xx) - uh(x))**2
        e = self.integral(f, celltype=celltype)
        if celltype is False:
            return np.sqrt(e.sum())
        else:
            return np.sqrt(e)
        return 

    def H1_semi_error(self, gu, guh, celltype=False):
        def f(x):
            xx = self.mesh.bc_to_point(x)
            xx, _ = self.mesh.surface.project(xx)
            return (gu(xx) - guh(x))**2
        e = self.integral(f, celltype=celltype)
        if celltype is False:
            return np.sqrt(e.sum())
        else:
            return np.sqrt(e)
        return 
#
#
#        qf = self.integrator  
#        bcs, ws = qf.quadpts, qf.weights
#        pp = mesh.bc_to_point(bcs)
#
#        if barycenter is True:
#            val0 = uh(bcs)
#        else:
#            val0 = uh(pp)
#        
#        if surface is not None:
#            Jp, grad = mesh.jacobi_matrix(bcs)
#            Jsp = surface.jacobi_matrix(pp)
#        val1 = u(pp)
#
#        if surface is not None:
#            Js = np.einsum('...ijk, ...imk->...imj', Jsp, Jp)
#            Gp = np.einsum('...ijk, ...imk->...ijm', Jp, Jp)
#            Gp = np.linalg.inv(Gp)
#            val1 = np.einsum('...ikj, ...ij->...ik', Js, val1)
#            val1 = np.einsum('...ikj, ...ij->...ik', Gp, val1)
#            val1 = np.einsum('...ijk, ...ij->...ik', Jp, val1)
#        
#        e = (val1 - val0)**2
#        axis = tuple(range(2, len(e.shape)))
#        e = np.sum(e, axis=axis)
#        e = np.einsum('i, ij->j', ws, e)
#        e *= self.area 
#        return np.sqrt(e.sum()) 

    def l2_error(self, u, uh):
        e = self.integral(u, barycenter=False)/np.sum(self.area)
        uI = uh.space.interpolation(u)
        gdof = uh.space.number_of_global_dofs()
        return np.sqrt(np.sum((uI - uh)**2)/gdof)

    def infty_error(self, u, uh):
        uI = uh.space.interpolation(u)
        gdof = uh.space.number_of_global_dofs()
        return np.max(np.abs(uI - uh))
