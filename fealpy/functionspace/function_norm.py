import numpy as np
import types

class FunctionNorm():
    def __init__(self, integrator, area=None):
        self.integrator = integrator
        self.area = area

    def L2_norm(u, mesh, funtype='scalar'):
        qf = self.integrator  
        bcs, ws = qf.quadpts, qf.weights
        if isinstance(u, types.FunctionType) or isinstance(u, types.MethodType):
            pp = mesh.bc_to_point(bcs)
            val = u(pp)
        else:
            val = u(bcs)
        if funtype is 'scalar':
            e = np.einsum('i, ij->j', ws, val)))
        elif funtype is 'vector':
            l = np.sum(val**2, axis=-1)
            e = np.einsum('i, ij->j', ws, l)
        else:
            raise ValueError('funtype "{}"'.format(funtype))

        if self.area is None:
            e *= mesh.area()
        else:
            e *= self.area
        return np.sqrt(e.sum()) 

    def L2_error(self, u, uh, funtype='scalar'):
        mesh = uh.V.mesh
        qf = self.integrator  
        bcs, ws = qf.quadpts, qf.weights
        val0 = uh.value(bcs)

        pp = mesh.bc_to_point(bcs)
        val1 = u(pp)
        if funtype is 'scalar':
            e = np.einsum('i, ij->j', ws, (val1-val0)**2)))
        elif funtype is 'vector':
            l = np.sum((val1 - val0)**2, axis=-1)
            e = np.einsum('i, ij->j', ws, l)
        else:
            raise ValueError('funtype "{}"'.format(funtype))

        if self.area is None:
            e *= mesh.area()
        else:
            e *= self.area
        return np.sqrt(e.sum()) 

    def l2_error(self, u, uh):
        uI = uh.V.interpolation(u)
        gdof = uh.V.number_of_global_dofs()
        return np.sqrt(np.sum((uI - uh)**2)/gdof)

    def H1_semi_error(self, gu, uh, funtype='scalar'):
        mesh = uh.V.mesh
        qf = self.integrator 
        bcs, ws = qf.quadpts, qf.weights
        if funtype is 'scalar':
            guh = uh.value(bcs)
        elif funtype is 'vector':
            guh = uh.grad_value(bcs)
        else:
            raise ValueError('funtype "{}"'.format(funtype))

        pp = mesh.bc_to_point(bcs)
        gu = gu(pp)
        e = np.sum((guh - gu)**2, axis=-1)
        e = np.einsum('i, ij->j', ws, e)
        if self.area is None:
            e *= mesh.area()
        else:
            e *= self.area
        return np.sqrt(e.sum()) 

    def H1_error(self, u, gu, uh, funtype='scalar'):
        e0 = self.L2_error(u, uh, funtype=funtype)
        e1 = self.H1_semi_error(gu, uh, funtype=funtype)
        return np.sqrt(e0**2 + e1**2)

    def div_error(self, divu, guh, funtype='scalar'):
        mesh = uh.V.mesh
        qf = self.integrator 
        bcs, ws = qf.quadpts, qf.weights
        val0 = guh.div_value(bcs)
        pp = mesh.bc_to_point(bcs)
        val1 = divu(pp)

        if funtype is 'scalar':
            e = np.einsum('i, ij->j', ws, (val1-val0)**2)))
        elif funtype is 'vector':
            l = np.sum((val1 - val0)**2, axis=-1)
            e = np.einsum('i, ij->j', ws, l)
        else:
            raise ValueError('funtype "{}"'.format(funtype))
        e = np.einsum('i, ij->j', ws, (val1 - val0)**2)
        if self.area is None:
            e *= mesh.area()
        else:
            e *= self.area
        return np.sqrt(e.sum()) 
