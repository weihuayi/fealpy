from . import PREProcessor
from .config import *
from ..decorator import variantmethod
from .tool import matrix_absA


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

    def grad_recovery(self):
        """
        pointwise Hessian of the solution
        
        Returns
            TensorLike: Hessian of solution
        """
        cm = self.cm
        ws = self.ws
        sm = self.sm
        Ndof = self.pspace.number_of_global_dofs()
        NN = self.NN
        if NN != Ndof:
            raise ValueError("The number of nodes is not equal to the number of dofs.")
        
        guh_incell = self._grad_uh() 
        c_guh = bm.einsum('cqd,c,q->cd',guh_incell,cm,ws)
        guh_node = bm.zeros((NN,self.GD),**self.kwargs0)
        guh_node = bm.index_add(guh_node, self.cell2dof, c_guh[:,None,...])
        guh_node /= sm[:,None]
        return guh_node
    
    def hessian_recovery(self):
        """
        pointwise Hessian of the solution
        
        Returns
            TensorLike: Hessian of solution
        """
        guh_node = self.grad_recovery()  # (NN, GD)
        
        pspace = self.pspace
        pcell2dof = self.pcell2dof            # (NC, ldof)
        gphi = pspace.grad_basis(self.bcs)    # (NC, NQ, ldof, GD)
        hess_incell = bm.einsum('cqid, cil -> cqld', gphi, guh_node[pcell2dof])

        cm = self.cm            # (NC,)
        ws = self.ws            # (NQ,)
        sm = self.sm            # (NN,)
        c_hess = bm.einsum('cqld, c, q -> cld', hess_incell, cm, ws)  # (NC, GD, GD)

        huh_node = bm.zeros((self.NN, self.GD, self.GD), **self.kwargs0)
        huh_node = bm.index_add(huh_node, self.cell2dof, c_hess[:, None, ...])  # (NN, GD, GD)
        huh_node /= sm[:, None, None]
        BdNodeidx = self.BdNodeidx
        huh_node = bm.set_at(huh_node, BdNodeidx, 0)
        return huh_node
    
    @variantmethod('arc_length')
    def monitor(self):
        """
        arc length monitor
        """
        guh_incell = self._grad_uh()
        self.M = bm.sqrt(1 +  self.beta * bm.sum(guh_incell**2,axis=-1))
        
    @monitor.register('matrix_arc_length')
    def monitor(self):
        """
        matrix arc length monitor
        """
        guh_incell = self._grad_uh()
        self.M = bm.sqrt(1 +  self.beta * bm.sum(guh_incell**2,axis=-1)) # NC,NQ
        self.M = self.M[...,None,None]*bm.eye(self.GD,**self.kwargs0) # NC,NQ,GD,GD
        
    @monitor.register('scaled_arc_length')
    def monitor(self):
        """
        scaled arc length monitor
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

    @monitor.register('linear_int_error')
    def monitor(self):
        """
        linear interpolation error monitor method
        """
        huh_node = self.hessian_recovery()  # (NN, GD, GD)
        huh_node = matrix_absA(huh_node, symmetric=True)  # (NN, GD, GD)
        d = self.GD
        cm = self.cm
        area = bm.sum(cm)
        
        H = bm.mean(huh_node[self.cell],axis=1)  # NC,GD,GD
        H_K = bm.trace(H,axis1= -2,axis2 = -1)**(2*d/(d+4))  # NC
        b = (bm.sum(cm*H_K)/(2*area))**((d+4)/(2*d))
        
        a = 0.001
        alpha = 0.5*(a+b)
        I = bm.eye(d,**self.kwargs0)
        II = bm.zeros((self.NC,d,d),**self.kwargs0)
        II += I[None,...]
        while (b - a > 0.001 * (a+b)*0.5):
            alpha = 0.5*(a+b)
            detH = bm.linalg.det(1/alpha * H + II)**(2/(d+4))
            ff = bm.sum(cm*detH) - 2*area
            if ff > 0:
                a = alpha
            else:
                b = alpha
        I_nv = bm.eye(d,**self.kwargs0)[None,...]
        II_nv = bm.repeat(I_nv, self.NN, axis=0)
        Hessian = II_nv + 1/alpha * huh_node  # NN,GD,GD
        detHess = bm.linalg.det(Hessian)**(-1/(d+4))
        M = detHess[...,None,None]*Hessian
        self.M = bm.mean(M[self.cell],axis=1)  # NC,GD,GD
    
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

    @mol_method.register('constant_projector')
    def mol_method(self):
        """
        constant projector mollification method
        """
        M = self.M # NC
        exp_nd = M.ndim - 2
        cell = self.cell  # NC,ldof
        sm = self.sm
        cm = self.cm
        shape = (self.NN,) + (self.GD,) * exp_nd
        M_node = bm.zeros(shape, **self.kwargs0)
        exp_sm = sm[(...,) + (None,) * exp_nd]
        exp_cm = cm[(...,) + (None,) * exp_nd]
        M_incell = bm.mean(M,axis=1) # (NC,...)
        for i in range(self.mol_times):
            if i != 0:
                M_incell = bm.mean(M_node[cell],axis=1)
            M_node.fill(0)
            M_node = bm.index_add(M_node, cell, M_incell[:,None,...]*exp_cm[:,None])
            M_node /= exp_sm

        self.M = M_incell
        self.M_node = M_node
    
    @mol_method.register('huangs_method')
    def mol_method(self):
        """
        Huang's method mollification method,only for matrix monitor
        """
        M = self.M
        GD = self.GD
        if M.ndim != 3:
            M = bm.mean(M,axis=1)  # NC,GD,GD
        cell = self.cell
        counts = bm.zeros((self.NN,), **self.kwargs0)
        ones = bm.ones((self.NC,self.GD+1), **self.kwargs0)
        counts = bm.index_add(counts, cell, ones)
        M_node = bm.zeros((self.NN, self.GD, self.GD), **self.kwargs0)
        
        M_node = bm.index_add(M_node, cell, M[:,None,...])
        M_node /= counts[:,None,None]

        for i in range(self.mol_times):
            M1 = M_node[cell]  # NC,GD+1,GD,GD
            wk = bm.sum(M1,axis=1)/(2*GD)  # NC,GD,GD
            M1 = wk[:,None,...] + M1 * (GD-1)/(2*GD) # NC,GD+1,GD,GD

            MM = bm.zeros((self.NN, GD, GD), **self.kwargs0)
            MM = bm.index_add(MM, cell, M1)
            MM /= counts[:,None,None]
            M_node = MM
        self.M = bm.mean(M_node[cell],axis=1)
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
        
    
