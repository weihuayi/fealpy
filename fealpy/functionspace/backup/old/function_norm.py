import numpy as np
import types

class FunctionNorm():
    def __init__(self, integrator, area):
        self.integrator = integrator
        self.area = area

    def integral(self, u, mesh, barycenter=False, elemtype=False):
        qf = self.integrator  
        bcs, ws = qf.quadpts, qf.weights
        if barycenter is False:
            pp = mesh.bc_to_point(bcs)
            val = u(pp)
        else:
            val = u(bcs)
        e = np.einsum('i, ij->j', ws, val)
        e *= self.area 
        if elemtype is True:
            return e
        else:
            return e.sum() 


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

    def L2_error(self, u, uh, mesh, barycenter=True, surface=None):

        qf = self.integrator  
        bcs, ws = qf.quadpts, qf.weights
        pp = mesh.bc_to_point(bcs)

        if barycenter is True:
            val0 = uh(bcs)
        else:
            val0 = uh(pp)
        
        if surface is not None:
            Jp, grad = mesh.jacobi_matrix(bcs)
            Jsp = surface.jacobi_matrix(pp)
        val1 = u(pp)

        if surface is not None:
            Js = np.einsum('...ijk, ...imk->...imj', Jsp, Jp)
            Gp = np.einsum('...ijk, ...imk->...ijm', Jp, Jp)
            Gp = np.linalg.inv(Gp)
            val1 = np.einsum('...ikj, ...ij->...ik', Js, val1)
            val1 = np.einsum('...ikj, ...ij->...ik', Gp, val1)
            val1 = np.einsum('...ijk, ...ij->...ik', Jp, val1)
        
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

class SurfaceFunctionNorm():
    def __init__(self, integrator, area):
        self.integrator = integrator
        self.area = area

    def L2_error(self):
        V = self.V
        mesh = V.mesh
        model = self.model
        
        qf = self.integrator 
        bcs, ws = qf.quadpts, qf.weights 
        pp = mesh.bc_to_point(bcs)
        n, ps = mesh.normal(bcs)
        l = np.sqrt(np.sum(n**2, axis=-1))
        area = np.einsum('i, ij->j', ws, l)/2.0

        val0 = self.uh.value(bcs)
        val1 = model.solution(ps)
        e = np.einsum('i, ij->j', ws, (val1 - val0)**2)
        e *= area
        return np.sqrt(e.sum()) 

    def H1_error(self):
        V = self.V
        mesh = V.mesh
        model = self.model
        
        qf = self.integrator
        bcs, ws = qf.quadpts, qf.weights 
        pp = mesh.bc_to_point(bcs)

        val0, ps, n= V.grad_value_on_surface(self.uh, bcs)
        val1 = model.gradient(ps)
        l = np.sqrt(np.sum(n**2, axis=-1))
        area = np.einsum('i, ij->j', ws, l)/2.0
        e = np.sum((val1 - val0)**2, axis=-1)
        e = np.einsum('i, ij->j', ws, e)
        e *=self.area
        return np.sqrt(e.sum()) 
