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

    def grad_recovery(self , use_projection=False):
        """
        pointwise Hessian of the solution
        
        Returns
            TensorLike: Hessian of solution
        """
        if not use_projection:
            # 原实现
            cm, ws, sm = self.cm, self.ws, self.sm
            guh_incell = self._grad_uh() 
            c_guh = bm.einsum('cqd,c,q->cd', guh_incell, cm, ws)
            guh_node = bm.zeros((self.NN, self.GD), **self.kwargs0)
            guh_node = bm.index_add(guh_node, self.cell, c_guh[:, None, ...])
            guh_node /= sm[:, None]
            return guh_node
        
        self._mass_gererator()
        pspace = self.pspace
        gphi = pspace.grad_basis(self.bcs)          # (NC, NQ, ldof, GD)
        guh_incell = self._grad_uh()                # (NC, NQ, GD)
        cm, ws = self.cm, self.ws                   # (NC,), (NQ,)
        # rhs_i = ∫ grad u · grad φ_i ≈ Σ_q w_q |K| (grad u · grad φ_i)
        rhs = bm.einsum('cqd,cqid,c,q->cid',
                guh_incell, gphi, cm, ws)   # (NC, ldof, GD)
        # 组装到全局（向量化到 dof 维度）
        Ndof = pspace.number_of_global_dofs()
        rhs_g = bm.zeros((Ndof, self.GD), **self.kwargs0)
        rhs_flat = rhs.reshape(-1, self.GD)                 # (NC*ldof, GD)
        idx_flat = self.pcell2dof.reshape(-1)               # (NC*ldof,)
        rhs_g = bm.index_add(rhs_g, idx_flat, rhs_flat)

        # 崩塌质量对角并防止除零；转换为后端数组避免稀疏矩阵类型
        M_lumped = bm.array(self.mass.sum(axis=1))
        M_lumped = bm.maximum(M_lumped, 1e-14)
        guh_dof = rhs_g / M_lumped[:, None]           # (Ndof, GD)

        return guh_dof
    
    def hessian_recovery(self):
        """
        pointwise Hessian of the solution
        
        Returns
            TensorLike: Hessian of solution
        """
        guh_node = self.grad_recovery(use_projection=False)
        
        pspace = self.pspace
        mimx = self.mesh.multi_index_matrix(p = pspace.p,etype=2) / pspace.p
        guh_node = bm.einsum('ij, cjd -> cid', mimx, guh_node[self.cell])
  
        gphi = pspace.grad_basis(self.bcs)    # (NC, NQ, ldof, GD)
        hess_incell = bm.einsum('cqid, cil -> cqld', gphi, guh_node)

        cm = self.cm            # (NC,)
        ws = self.ws            # (NQ,)
        sm = self.sm            # (NN,)
        c_hess = bm.einsum('cqld, c, q -> cld', hess_incell, cm, ws)  # (NC, GD, GD)

        huh_node = bm.zeros((self.NN, self.GD, self.GD), **self.kwargs0)
        huh_node = bm.index_add(huh_node, self.cell2dof, c_hess[:, None, ...])  # (NN, GD, GD)
        huh_node /= sm[:, None, None]
        huh_node = 0.5*(huh_node + bm.permute_dims(huh_node, axes=(0,2,1)))
        BdNodeidx = self.BdNodeidx
        huh_node = bm.set_at(huh_node, BdNodeidx, 0)
        
        # pspace = self.pspace
        # Ndof = pspace.number_of_global_dofs()
        # cell2dof = self.pcell2dof                     # (NC, ldof)
        # gphi = pspace.grad_basis(self.bcs)            # (NC, NQ, ldof, GD)
        # phi  = pspace.basis(self.bcs)                 # (NC, NQ, ldof)
        # cm, ws = self.cm, self.ws                     # (NC,), (NQ,)

        # # 单元上计算 Hessian（梯度的梯度）
        # hess_q = bm.einsum('cqli, clj -> cqji', gphi, guh_dof[cell2dof])   # (NC, NQ, GD, GD)
        # # L2 投影到自由度：rhs_l^{ij} = ∑_q w_q |K| hess_q^{ij} phi_l
        # rhs = bm.einsum('cqij, cql, c, q -> clij',
        #                 hess_q, phi, cm, ws)          # (NC, ldof, GD, GD)
        # # 装配到全局
        # rhs_g = bm.zeros((Ndof, self.GD, self.GD), **self.kwargs0)
        # rhs_g = bm.index_add(rhs_g, cell2dof, rhs)

        # # 崩塌质量阵对角（行和），防止除零
        # M_lumped = bm.array(self.mass.sum(axis=0)).reshape(-1)

        # huh_dof = rhs_g / M_lumped[:, None, None]     # (Ndof, GD, GD)
        # huh_dof = 0.5 * (huh_dof + bm.permute_dims(huh_dof, axes=(0, 2, 1)))
        # bddof_indx = self.pspace.is_boundary_dof()
        # huh_dof = bm.set_at(huh_dof, bddof_indx, 0)
        
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
        R_mean =(bm.einsum('q,cq,cq ->',self.ws,R+1, self.d*self.rm)/
                 bm.sum(self.cm))
        if bm.max(R_mean) < 1e-8:
            print("Warning: R_mean is too small, using default value.")
            R_mean = 1.0
        alpha = self.beta/((1.0-self.beta))
        lambda_1 = 1 + alpha*R # NC,NQ
        lambda_2 = 1/lambda_1
        self.M = lambda_1[...,None,None]*v[...,None,:]*v[...,None] + \
                 lambda_2[...,None,None]*v_orth[...,None,:]*v_orth[...,None] # NC,NQ,TD,TD

    @monitor.register('linear_int_error')
    def monitor(self):
        """
        linear interpolation error monitor method
        """
        huh_node = self.hessian_recovery()  # (Ndof, GD, GD)
        huh_node = matrix_absA(huh_node, symmetric=True)  # (Ndof, GD, GD)
        d = self.GD
        cm = self.cm
        area = bm.sum(cm)
        
        H = bm.mean(huh_node[self.cell],axis=1)  # NC,GD,GD
        trH = bm.trace(H, axis1=-2, axis2=-1)
        trH = bm.maximum(trH, 1e-14)
        H_K = trH ** (2 * d / (d + 4))  # (NC,)
        
        S = bm.sum(cm * H_K)
        denom = bm.maximum(2 * area, 1e-14)
        target = bm.maximum(S / denom, 1e-14)
        pwr = (d + 4) / (2 * d)
        b = target ** pwr
        
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
        alpha = 0.5 * (a + b)
        I_nv = bm.eye(d,**self.kwargs0)[None,...]
        # Ndof = self.pspace.number_of_global_dofs()
        II_nv = bm.repeat(I_nv, self.NN, axis=0)
        Hessian = II_nv + 1/alpha * huh_node  # Ndof,GD,GD
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
        
    
