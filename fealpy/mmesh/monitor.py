from . import PREProcessor
from .config import *
from ..decorator import variantmethod


class Monitor(PREProcessor):
    def __init__(self, mesh, beta, space, config: Config, **kwargs) -> None:
        super().__init__(mesh = mesh, space = space, config = config, **kwargs)
        self.beta = beta
        self.r = config.r
        self.alpha = config.alpha
        self.mol_times = config.mol_times
    
    def _grad_uh(self):
        """
        pointwise gradient of the solution
        
        Returns
            TensorLike: gradient of solution
        """
        uh = self.uh
        pspace = self.pspace
        pcell2dof = self.pcell2dof
        gphi = pspace.grad_basis(self.bcs) # change
        guh_incell = bm.einsum('cqid , ci -> cqd ',gphi,uh[pcell2dof])
        return guh_incell

    def _mp_grad_uh(self):
        """
        pointwise gradient of the solution for multiphysics
        
        Returns
            TensorLike: gradient of solution for multiphysics
        """
        uh = self.uh
        pspace = self.pspace
        pcell2dof = self.pcell2dof
        gphi = pspace.grad_basis(self.bcs)
        guh_incell = bm.einsum('cqid , cil -> cqld ',gphi,uh[pcell2dof])
        return guh_incell

    @variantmethod('arc_length')
    def monitor(self):
        """
        arc length monitor
        """
        guh_incell = self._grad_uh()
        self.M = bm.sqrt(1 +  self.beta * bm.sum(guh_incell**2,axis=-1))
        
    @monitor.register('norm_arc_length')
    def monitor(self):
        """
        normalized arc length monitor
        """
        guh_incell = self._grad_uh()
        R = bm.max(bm.linalg.norm(guh_incell,axis=-1))
        if R <= 1e-15:
            R = 1
        self.M = bm.sqrt(1 + self.beta * bm.sum(guh_incell**2,axis=-1)/R**2)
    
    @monitor.register('mp_arc_length')
    def monitor(self):
        """
        arc length monitor for multiphysics
        """
        guh_incell = self._mp_grad_uh()
        self.M = bm.sqrt(1 + 1/self.dim*bm.sum(self.beta[None,None,:]*
                                   bm.sum(guh_incell**2,axis=-1),axis=-1))

    @monitor.register('matrix_normal')
    def monitor(self):
        """
        matrix normal monitor method
        """
        guh_incell = self._grad_uh() # NC,NQ,TD
        
        norm_guh_cell = bm.linalg.norm(guh_incell,axis=-1) # NC,NQ
        is_zero = norm_guh_cell < 1e-15
        v = bm.zeros_like(guh_incell,**self.kwargs0)
        v = bm.set_at(v, is_zero, bm.array([1.0,0.0],**self.kwargs0))
        v = bm.set_at(v, ~is_zero, guh_incell[~is_zero]/norm_guh_cell[~is_zero][...,None])
        v_orth = bm.stack([-v[..., 1], v[..., 0]], axis=-1)

        R = bm.sqrt(1 +  bm.sum(guh_incell**2,axis=-1))-1 # NC,NQ
        R_mean =(bm.einsum('q,cq,cq ->',self.ws,R, self.d*self.rm)/
                 bm.sum(self.cm))
        if bm.max(R_mean) < 1e-15:
            print("Warning: R_mean is too small, using default value.")
            R_mean = 1.0
        alpha = self.beta/(R_mean*(1.0-self.beta))
        lambda_1 = 1 + alpha*R # NC,NQ
        lambda_2 = 1/lambda_1
        self.M = lambda_1[...,None,None]*v[...,None,:]*v[...,None] + \
                 lambda_2[...,None,None]*v_orth[...,None,:]*v_orth[...,None] # NC,NQ,TD,TD

    @variantmethod('projector')
    def mol_method(self):
        """
        projection operator mollification method
        """
        M = self.M
        exp_nd = M.ndim - 2
        cell2dof = self.cell2dof  # NC,NQ
        sm = self.sm
        d = self.d # NC,NQ
        rm = self.rm
        exp_sm = sm[(...,) + (None,) * exp_nd]
        shape = (self.NN,) + (self.TD,) * exp_nd
        phi = self.mspace.basis(self.bcs)
        dphi = phi*rm*d[(...,) + (None,) *(3-d.ndim)]  # NC,NQ,...
        M = M*rm*d[(...,) + (None,) * (2-d.ndim+exp_nd)]  # NC,NQ,...
        M_node = bm.zeros(shape, **self.kwargs0)
        for i in range(self.mol_times):
            if i != 0:
                M = bm.einsum('cqi,ci...->cq...', dphi, M_node[cell2dof])
            M_incell = bm.einsum('cq...,q-> c...',M,self.ws)
            M_node.fill(0)
            M_node = bm.index_add(M_node, cell2dof, M_incell[:,None,...])
            M_node /= exp_sm
            
        self.M = bm.einsum('cqi,ci...->cq...', phi, M_node[cell2dof])
        self.M_node = M_node
    
    @mol_method.register('heatequ')
    def mol_method(self):
        """
        heat equation mollification method
        """
        M = self.M
        h = self.hmin
        r = self.r
        R = r*(1+r)
        dt = 1/self.mol_times
        mass = self.mass
        pspace = self.pspace
        bform = BilinearForm(pspace)
        lform = LinearForm(pspace)
        SDI = self.SDI
        SSI = self.SSI
        
        bform.add_integrator(SDI)
        lform.add_integrator(SSI)
        SDI.coef = h**2*R*dt
        SDI.clear()
        M_bar = pspace.function()
        A = bform.assembly() + mass
        for i in range(self.mol_times):
            SSI.source = M
            SSI.clear()
            b = lform.assembly()
            M_bar[:] = cg(A,b,atol=1e-5,returninfo=True)[0]
            M = M_bar(self.bcs)
        self.M = M
        self.M_node = M_bar[:]