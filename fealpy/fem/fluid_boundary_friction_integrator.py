from typing import Optional, Union, Callable
from scipy.sparse import csr_matrix,hstack,vstack,spdiags,bmat
import numpy as np


class FluidBoundaryFrictionIntegrator:

    def __init__(self, mu, threshold=None, q=None):
        """
        @brief 
        (mu \nabla u \cdot n, v)_{\partial \Omega}
        @param[in] mu 
        """
        self.mu = mu 
        self.q = q
        self.threshold = threshold

    def assembly_face_matrix(self, space, out=None):
        """
        @brief 组装面元向量
        """
        if isinstance(space, tuple) and ~isinstance(space[0], tuple):
            return self.assembly_face_matrix_for_vspace_with_scalar_basis(
                    space, out=out)
        else:
            return self.assembly_face_matrix_for_vspace_with_vector_basis(
                    space, out=out)

    def assembly_face_matrix_for_vspace_with_scalar_basis(self, space, out=None):
        """
        """
        assert isinstance(space, tuple) and ~isinstance(space[0], tuple) 
        
        mu = self.mu
        mesh = space[0].mesh # 获取网格对像

        if isinstance(self.threshold, np.ndarray):
            index = self.threshold
        else:
            index = mesh.ds.boundary_face_index()
            if callable(self.threshold):
                bc = mesh.entity_barycenter('face', index=index)
                index = index[self.threshold(bc)]

        edge2cell = mesh.ds.edge2cell[index]
        emeasure = mesh.entity_measure('face', index=index)
        gdof = space[0].number_of_global_dofs()
        face2dof = space[0].face_to_dof()[index]
        cell2dof = space[0].cell_to_dof()
        #边积分  
        q = self.q if self.q is not None else space[0].p + 3
        qf = mesh.integrator(q, 'face')
        ebcs, ews = qf.get_quadrature_points_and_weights()
        #边基函数 
        ephi = space[0].face_basis(ebcs, index)
        egphi = space[0].edge_grad_basis(ebcs, edge2cell[:,0], edge2cell[:,2])
        n = mesh.face_unit_normal(index=index)
        
        if callable(mu):
            pgx0 = np.einsum('ij,i,ijk,jim,j,j->jkm',mu(ebcs,index), ews,ephi,egphi[...,0],n[:,0],emeasure)
            pgy1 = np.einsum('ij,i,ijk,jim,j,j->jkm',mu(ebcs,index), ews,ephi,egphi[...,1],n[:,1],emeasure)
            pgx1 = np.einsum('ij,i,ijk,jim,j,j->jkm',mu(ebcs,index), ews,ephi,egphi[...,0],n[:,1],emeasure)
            pgy0 = np.einsum('ij,i,ijk,jim,j,j->jkm',mu(ebcs,index), ews,ephi,egphi[...,1],n[:,0],emeasure)  
           
        else:
            pgx0 = mu*np.einsum('i,ijk,jim,j,j->jkm',ews,ephi,egphi[...,0],n[:,0],emeasure)
            pgy1 = mu*np.einsum('i,ijk,jim,j,j->jkm',ews,ephi,egphi[...,1],n[:,1],emeasure)
            pgx1 = mu*np.einsum('i,ijk,jim,j,j->jkm',ews,ephi,egphi[...,0],n[:,1],emeasure)
            pgy0 = mu*np.einsum('i,ijk,jim,j,j->jkm',ews,ephi,egphi[...,1],n[:,0],emeasure)  
           

        J1 = np.broadcast_to(face2dof[:,:,None],shape =pgx0.shape) 
        tag = edge2cell[:,0]
        I1 = np.broadcast_to(cell2dof[tag][:,None,:],shape = pgx0.shape)
        
        D00 = csr_matrix((pgx0.flat,(J1.flat,I1.flat)),shape=(gdof,gdof))
        D11 = csr_matrix((pgy1.flat,(J1.flat,I1.flat)),shape=(gdof,gdof))
        D10 = csr_matrix((pgy0.flat,(J1.flat,I1.flat)),shape=(gdof,gdof))
        D01 = csr_matrix((pgx1.flat,(J1.flat,I1.flat)),shape=(gdof,gdof))
        
        r = bmat([[D00, D01],[D10, D11]])
        if space[0].doforder == 'vdims':
            row = r.shape[0]
            result = np.zeors_like(r)
            result[::2,:] = result[:row//2,:] 
            result[1::2,:] = result[row//2:,:] 
            r =  result
        if out is None:
            return r
        else:
            out += r

