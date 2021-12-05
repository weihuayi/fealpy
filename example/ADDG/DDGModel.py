import numpy as np
from math import sqrt
from Lagrange_fem_space import LagrangeFiniteElementSpace
from timeit import default_timer as timer
from scipy.sparse.linalg import spsolve

class PoissonDDGModel(object):
    def __init__(self, pde, mesh, qf, IntervalQuadrature,p):
        self.space = LagrangeFiniteElementSpace(mesh, p, spacetype='D') 
        self.mesh = mesh
        self.pde = pde 
        self.qf = qf
        self.uh = self.space.function()
        self.uI = self.space.interpolation(pde.solution)
        self.cellmeasure = mesh.entity_measure('cell')
        self.IntervalQuadrature = IntervalQuadrature
        self.cellidx = np.arange(mesh.number_of_cells())
        self.pp = self.space.bc_to_point_cell(self.qf.quadpts)
        self.u = self.pde.solution(self.pp)
        self.cellidx = np.arange(self.mesh.number_of_cells())
        self.p = p
        
        
    def get_left_matrix(self,beta1,beta2,k,index,Cdtype):
        A = self.space.DDG_stiff_matrix(self.qf, self.pde.diffusion_coefficient)
        P,S,H = self.space.edge_matrix(self.IntervalQuadrature, self.pde.diffusion_coefficient,index,Cdtype)
        AD = A-S-S.T+beta1*P + beta2*H
        return AD
    
    def get_source_vector(self):
        return self.space.DDG_source_vector(self.pde.source, self.qf)
    
    def get_right_vector(self,beta1,idx,cfun,Cdtype):
        ws = self.IntervalQuadrature.weights
        g0 = self.pde.dirichlet
        mesh = self.mesh
        space = self.space
        index = mesh.ds.boundary_edge_index() 
        edge2cell = mesh.ds.edge_to_cell()
        n = mesh.edge_unit_normal()
        length = mesh.entity_measure('edge')
        
        pp = space.bc_to_point_edge(self.IntervalQuadrature)
        g_value = g0(pp)
       
        cellidx_0 = edge2cell[:,0]
        
        phi = space.Hat_function(pp,cellidx_0)
        gphi =  space.Hat_Gradients(pp,cellidx_0)
        d = cfun(pp)
        we = space.We(self.IntervalQuadrature,cfun,idx,Cdtype)
        dgphi = np.einsum('...i, ...imn->...imn', d, gphi)
        if Cdtype =='D':
            bcs = np.array([1/3,1/3,1/3])
            ps = space.bc_to_point_cell(bcs)
            d = cfun(ps)
            dgphi = np.einsum('...i, ...imn->...imn', d[cellidx_0], gphi)
        P_bd = np.einsum('t, tm, tmk,tm->mk', ws, g_value[:,index], phi[:,index],we[:,index])
        S_bd = np.einsum('t, tm,tmkq,mq,m->mk',ws, g_value[:,index], dgphi[:,index], n[index], length[index])
       
        cell2dof = space.cell_to_dof()
        dof1  = cell2dof[edge2cell[:,0]]
        dof = dof1[index]
        gdof = space.number_of_global_dofs()
        P_bd = np.bincount(dof.flat, weights=P_bd.flat, minlength=gdof).reshape(-1)
        S_bd = 1/2*np.bincount(dof.flat, weights=S_bd.flat, minlength=gdof).reshape(-1)
        b = self.get_source_vector()
        f = b + beta1*P_bd -S_bd.T
        return f 
    
    def solve(self,beta1,beta2,index,Cdtype):
        start = timer()
        AD = self.get_left_matrix(beta1,beta2,self.pde.diffusion_coefficient,index,Cdtype)
        b = self.get_right_vector(beta1,index,self.pde.diffusion_coefficient,Cdtype)
        end = timer()
        print("Construct linear system time:", end - start)
        start = timer()
        self.uh[:] = spsolve(AD, b)
        end = timer()
        
        print("Solve time:", end-start)
        
        ls = {'AD':AD, 'b':b, 'solution':self.uh.copy()}

        return ls # return the linear system
    
    def get_DG_error(self,uh,index,cfun = None):
        ws = self.qf.weights
        bcs = self.qf.quadpts
        ws2 = self.IntervalQuadrature.weights
        mesh = self.mesh
        cell2dof = self.space.dof.cell2dof
        
        edge2cell = mesh.ds.edge_to_cell()
        lidx1 = edge2cell[:,0]
        lidx2 = edge2cell[:,1]
        index3 = mesh.ds.boundary_edge_index()
        index4 = mesh.ds.inter_edge_index()
        uh2cell = uh[cell2dof]
        uh1 = uh2cell[lidx1]
        uh2 = uh2cell[lidx2]
        new_uh = np.c_[uh1,uh2]
        
        guh  = self.space.grad_value(self.uh, bcs)
        pp = self.space.bc_to_point_cell(bcs)
        gu = self.pde.gradient(pp)
        
        p = self.space.bc_to_point_edge(self.IntervalQuadrature)
        phi1 = self.space.Hat_function(p,lidx1)
        phi2 = self.space.Hat_function(p,lidx2)
        phi = np.append(phi1,-phi2,axis = 2)
        uh_jump = np.einsum('tmj,mj->mt', phi,new_uh)
        uh_jump[index3] = 0*uh_jump[index3]
        
        uh_D = np.einsum('mj,tmj->mt', uh1,phi1)
        gD = self.pde.solution(p)
        gD = gD.swapaxes(0, 1)
        u = gD-uh_D
        u[index4] = 0*u[index4]
        if cfun is not None:
            a = self.pde.diffusion_coefficient(pp)
            a_K = np.einsum('t, tm->m', ws, a)
            we = self.space.We(cfun,index)
            error1 = np.einsum('k, kmj,m,m->',ws, (guh - gu)**2,self.cellmeasure,a_K)
            error2 = np.einsum('t,mt,m->',ws2, uh_jump**2,we)
            error3 = np.einsum('t,mt,m->', ws2,u**2,we)
        else:
            error1 = np.einsum('k, kmj,m->',ws, (guh - gu)**2,self.cellmeasure)
            error2 = np.einsum('t,mt->',ws2, uh_jump**2)
            error3 = np.einsum('t,mt->', ws2,u**2)
        error = error1+error2+error3
        return sqrt(error)
    
    def get_rel_error(self,uh,indx,cfun = None):
        ws = self.qf.weights
        bcs = self.qf.quadpts
        pp = self.space.bc_to_point_cell(bcs)
        gu = self.pde.gradient(pp)
        error1 = self.get_DG_error(uh,cfun)
        if cfun is not None:
            a = self.pde.diffusion_coefficient(pp)
            a_K = np.einsum('t, tm->m', ws, a)
            error2 = np.einsum('k, kmj,m,m->',ws, gu**2,self.cellmeasure,a_K)
        else:
            error2 = np.einsum('k, kmj,m->',ws, gu**2,self.cellmeasure)
        error = error1/sqrt(error2)
        return error
    
    
    