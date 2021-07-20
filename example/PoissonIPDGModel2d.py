import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.quadrature import GaussLegendreQuadrature
from math import sqrt
from fealpy.solver import solve
from scipy.sparse.linalg import spsolve
from timeit import default_timer as timer



class PoissonIPDGModel2d(object):
    def __init__(self, pde, mesh, p):
        self.space = ScaledMonomialSpace2d(mesh, p)
        self.mesh = mesh
        self.pde = pde 
        self.cellbarycenter = mesh.entity_barycenter('cell')
        self.p = p
        self.cellmeasure = mesh.entity_measure('cell')
    
    def Jump(self):
        mesh = self.mesh
        space = self.space
        edge2cell = mesh.ds.edge_to_cell()
        qf = GaussLegendreQuadrature(self.p + 4)
        isBdEdge = (edge2cell[:, 0] == edge2cell[:, 1])
        bcs = qf.quadpts
        ps = mesh.edge_bc_to_point(bcs)
        
        #Construct the jump of edge
        phi0 = space.basis(ps, index=edge2cell[:, 0]) # (NQ, NE, ldof)
        phi1 = space.basis(ps, index=edge2cell[:, 1]) # (NQ, NE, ldof)
        phi1[:,isBdEdge] = 0*phi1[:,isBdEdge]
        jump = np.append(phi0,-phi1,axis = 2) # (NQ, NE, 2*ldof)
        return jump
    
    def Grad_average(self):
        p =self.p
        mesh = self.mesh
        space = self.space
        edge2cell = mesh.ds.edge_to_cell()
        isBdEdge = (edge2cell[:, 0] == edge2cell[:, 1])
        qf = GaussLegendreQuadrature(p + 4)
        bcs = qf.quadpts
        ps = mesh.edge_bc_to_point(bcs)
        
        #Construct the grad_average of edge
        gphi0 = space.grad_basis(ps, edge2cell[:, 0])
        gphi1 = space.grad_basis(ps,edge2cell[:, 1])
        gphi0[:,isBdEdge] = 2*gphi0[:,isBdEdge]
        gphi1[:,isBdEdge] = 0*gphi1[:,isBdEdge]
        gphi = 1/2*np.append(gphi0,gphi1,axis = 2)
        return gphi
    
    def penalty_matrix(self):
        p = self.p
        mesh = self.mesh
        space = self.space
        edge2cell = mesh.ds.edge_to_cell()
        qf = GaussLegendreQuadrature(p + 4)
        ws = qf.weights
        jump = self.Jump()
        
        # Construct the penalty matrix of computing element
        P = np.einsum('i, ijk, ijm->jkm', ws, jump, jump, optimize=True)
        
        # Construct the global penalty matrix
        ldof = space.number_of_local_dofs(p=p, doftype='cell')
        cell2dof = space.cell_to_dof()
        dof1  = cell2dof[edge2cell[:,0]]
        dof2  = cell2dof[edge2cell[:,1]]
        dof = np.append(dof1,dof2,axis = 1)
        I = np.einsum('k, ij->ijk', np.ones(2*ldof), dof)
        J = I.swapaxes(-1, -2)
        gdof = space.number_of_global_dofs(p=p)
        P = csr_matrix((P.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return P
    
    def flux_matrix(self):
        p =self.p
        mesh = self.mesh
        space = self.space
        edge2cell = mesh.ds.edge_to_cell()
        qf = GaussLegendreQuadrature(p + 4)
        ws = qf.weights
        eh = mesh.entity_measure('edge')
        
        jump = self.Jump()
        gphi = self.Grad_average()
        n = mesh.edge_unit_normal()
        
        # Construct the flux matrix of computing element
        S = np.einsum('i, ijk, ijmp, jp, j->jkm', ws, jump, gphi, n, eh)
        
        # Construct the global flux matrix
        ldof = space.number_of_local_dofs(p=p, doftype='cell')
        cell2dof = space.cell_to_dof()
        dof1  = cell2dof[edge2cell[:,0]]
        dof2  = cell2dof[edge2cell[:,1]]
        dof = np.append(dof1,dof2,axis = 1)
        I = np.einsum('k, ij->ijk', np.ones(2*ldof), dof)
        J = I.swapaxes(-1, -2)
        gdof = space.number_of_global_dofs(p=p)
        S = csr_matrix((S.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return S
    
    def triangle_measure(self, tri):
        v1 = tri[1] - tri[0]
        v2 = tri[2] - tri[0]
        area = np.cross(v1, v2)/2
        return area
    
    def source_vector(self, f, celltype=False, q=None):
        p = self.p
        space = self.space
        mesh = self.mesh
        node = mesh.node
        bc = space.cellbarycenter
        edge = mesh.entity('edge')
        cell2dof = space.cell_to_dof()
        edge2cell = mesh.ds.edge_to_cell()
        qf = mesh.integrator(p+4)
        bcs, ws = qf.quadpts, qf.weights
        tri_0 = [bc[edge2cell[:, 0]], node[edge[:, 0]], node[edge[:, 1]]]
        a_0 = self.triangle_measure(tri_0)#NE
        pp_0 = np.einsum('ij, jkm->ikm', bcs, tri_0)#每个三角形中高斯积分点对应的笛卡尔坐标点
        fval_0 = f(pp_0)
        phi_0 = space.basis(pp_0, edge2cell[:, 0])
        gdof = space.number_of_global_dofs(p=p)
        F = np.zeros(gdof, dtype=np.float64)
        bb_0 = np.einsum('i, ij, ijk,j->jk', ws, fval_0, phi_0, a_0)
        np.add.at(F, cell2dof[edge2cell[:, 0]], bb_0)
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        if np.sum(isInEdge) > 0:
            tri_1 = [
                    bc[edge2cell[isInEdge, 1]],
                    node[edge[isInEdge, 1]],
                    node[edge[isInEdge, 0]]
                    ]
            a_1 = self.triangle_measure(tri_1)
            pp_1 = np.einsum('ij, jkm->ikm', bcs, tri_1)
            fval_1 = f(pp_1)
            phi_1 = space.basis(pp_1, edge2cell[isInEdge, 1])
            bb_1 = np.einsum('i, ij, ijk,j->jk', ws, fval_1, phi_1, a_1)
            np.add.at(F, cell2dof[edge2cell[isInEdge, 1]], bb_1)
        return F 
    
    def get_left_matrix(self,beta,alpha):
        A = self.space.stiff_matrix()
        S = self.flux_matrix()
        P = self.penalty_matrix()
        AD = A-S+alpha*S.T+beta*P
        return AD

    def get_right_vector(self):
        return self.source_vector(self.pde.source)
    
    def solve(self,beta,alpha):
        AD = self.get_left_matrix(beta,alpha)
        b = self.get_right_vector()
        self.uh = self.space.function()
        self.uh[:] = spsolve(AD, b)
        ls = {'A':AD, 'b':b, 'solution':self.uh.copy()}

        return ls # return the linear system

