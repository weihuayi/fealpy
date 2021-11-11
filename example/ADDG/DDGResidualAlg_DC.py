import numpy as np
from Lagrange_fem_space import LagrangeFiniteElementSpace
from IntervalQuadrature import IntervalQuadrature

class DDGResidualEstimators():
    def __init__(self,uh,mesh,pde,p):
        self.space = LagrangeFiniteElementSpace(mesh, p, spacetype='D') 
        self.mesh = mesh
        self.qf = mesh.integrator(p+3)
        self.intervalQuadrature = IntervalQuadrature(p+3)
        self.pde = pde 
        self.p = p
        self.uh = uh
        self.k = self.pde.diffusion_coefficient
        pass

    def barycenter(self):
        mesh = self.mesh
        node = mesh.node
        cell = mesh.ds.cell
        barycenter = 1/3*np.sum(node[cell],axis = 1)  #(NC,dim)
        return barycenter
    
    def diffusion_coefficient(self,cfun):
        p = self.barycenter()
        kval = cfun(p)
        return kval
    
    def diffusion_coefficient_average(self, cfun):
        edge2cell = self.mesh.ds.edge_to_cell()
        lidx1 = edge2cell[:,0]
        lidx2 = edge2cell[:,1]
        kval = self.diffusion_coefficient(cfun)
        mk = 1/2*(kval[lidx1]+kval[lidx2])
        return mk
    
    def edge_diffusion_coefficient(self,idx,cfun):
        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        lidx1 = edge2cell[:,0]
        lidx2 = edge2cell[:,1]
        kval = self.diffusion_coefficient(cfun)
        if idx == 1:
            we = 1/2*(kval[lidx1]+kval[lidx2])
        elif idx ==2:
            we = (2*kval[lidx1]*kval[lidx2])/(kval[lidx1]+kval[lidx2])
        elif idx ==3:
            we = np.sqrt(kval[lidx1]*kval[lidx2])
        return we
    
    def uh_jump(self):
        mesh = self.mesh
        space = self.space
        NC = mesh.number_of_cells()
        cell2dof = space.dof.cell2dof
        cell2edge = mesh.ds.cell_to_edge()
        edge2cell = mesh.ds.edge_to_cell()
        lidx1 = edge2cell[:,0]
        lidx2 = edge2cell[:,1]
        
        cell2edge2cell = edge2cell[cell2edge,0:2]
        uh2cell = self.uh[cell2dof]
        new_uh = uh2cell[cell2edge2cell,:].reshape(NC,3,-1)
        
        pp = space.bc_to_point_edge(self.intervalQuadrature)
        phi1 = space.Hat_function(pp,lidx1)
        phi2 = space.Hat_function(pp,lidx2)
        phi = np.append(phi1,-phi2,axis = 2)
        new_phi = phi[:,cell2edge]
        uh_jump = np.einsum('tmij,mij->tmi', new_phi,new_uh)
        return uh_jump
    
    def Rf_estimate(self,cfun):
        mesh = self.mesh
        space = self.space
        bcs, ws = self.qf.quadpts, self.qf.weights
        cell2edge = mesh.ds.cell_to_edge()
        cellmeasure = mesh.entity_measure('cell')
        length = mesh.edge_length(cell2edge.reshape(-1))
        length = length.reshape(-1,3)
        h_k = np.max(length,axis = 1)
        
        f = self.pde.source
        pp = space.bc_to_point_cell(bcs)
        fval = f(pp)
        fval = np.einsum('t, tm->m', ws, fval)
        L_uh = space.Laplace_uh(self.uh, self.qf)
        
        kval = self.diffusion_coefficient(cfun)
        D_uh = np.einsum('t, t->t', kval, L_uh)
        Rf_eta = (cellmeasure*h_k**2*(fval+D_uh)**2)/kval
        return Rf_eta
    
    def Ju_estimate(self,idx,cfun):
        mesh = self.mesh
        cell2edge = mesh.ds.cell_to_edge()
        ws = self.intervalQuadrature.weights
        index = mesh.ds.boundary_edge_index()
        kval = self.edge_diffusion_coefficient(idx,cfun)
        kval[index] = 0*kval[index]
        we = kval[cell2edge]
        uh_jump = self.uh_jump()
        Ju_eta = np.einsum('t,tmi,mi->m',ws, uh_jump**2,we)
        return Ju_eta
    
    def RD_estimate(self,idx,cfun):
        mesh = self.mesh
        space = self.space
        cell2dof = space.dof.cell2dof
        cell2edge = mesh.ds.cell_to_edge()
        edge2cell = mesh.ds.edge_to_cell()
        index = mesh.ds.inter_edge_index()
        lidx1 = edge2cell[:,0]
        kval = self.edge_diffusion_coefficient(idx,cfun)
        kval[index] = 0*kval[index]
        we = kval[cell2edge]
       
        ws = self.intervalQuadrature.weights
        pp = space.bc_to_point_edge(self.intervalQuadrature)
        phi = space.Hat_function(pp,lidx1)
        phi = phi[:,cell2edge]
        p = pp[:,cell2edge]
        gD = self.pde.solution(p)
        
        cell2edge2cell = edge2cell[cell2edge,0]
        uh2cell = self.uh[cell2dof]
        new_uh = uh2cell[cell2edge2cell]
        uh_D = np.einsum('mij,tmij->tmi', new_uh,phi)
        RD_estimate = np.einsum('t,tmi,mi->m', ws,(gD-uh_D)**2,we)
        return RD_estimate

    def Ji_estimate(self,idx,cfun):
        mesh = self.mesh
        space = self.space
        cell2dof = space.dof.cell2dof
        cell2edge = mesh.ds.cell_to_edge()
        edge2cell = mesh.ds.edge_to_cell()
        index = mesh.ds.boundary_edge_index()
        length = mesh.edge_length()
        length[index] = 0*length[index]
        ne = mesh.edge_unit_normal()
        length = length[cell2edge]
        
        ws = self.intervalQuadrature.weights
        lidx1 = edge2cell[:,0]
        lidx2 = edge2cell[:,1]
        
        pp = space.bc_to_point_edge(self.intervalQuadrature)
        G_Hat1 =  space.Hat_Gradients(pp,lidx1)
        G_Hat2 =  space.Hat_Gradients(pp,lidx2)
        uh2cell = self.uh[cell2dof]
       
        kval = self.diffusion_coefficient(cfun)
        
        k3 = 1/2*(kval[lidx1]+kval[lidx2])
        we = 1/k3[cell2edge]
        
        jump1 = np.einsum('m, mj, tmjk,mk ->tm',kval[lidx1], uh2cell[lidx1] , G_Hat1,ne)
        jump2 = np.einsum('m, mj, tmjk,mk ->tm',kval[lidx2], uh2cell[lidx2] , G_Hat2,ne)
        jump1 = jump1[:,cell2edge]
        jump2 = jump2[:,cell2edge]
        Ji_estimate =1/2*np.einsum('t,tmi,mi,mi->m', ws,(jump1-jump2)**2,length**2,we)
        return Ji_estimate
    
    
       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        