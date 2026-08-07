from typing import TypedDict, Callable, Tuple, Union, Optional

from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike 
from fealpy.sparse import COOTensor,CSRTensor
from fealpy.mesh import TriangleMesh

from opt import Problem
from .rropreconditioner import ProjectedCGPreconditioner

class TriSurfMeshProblem(Problem):
    def __init__(self, options:dict):
        self.mesh = options["mesh"]
        node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        localEdge = self.mesh.localEdge
        v0 = [node[cell[:,j],:] - node[cell[:,i],:] for i,j in localEdge]
        self.nv0 = bm.cross(v0[2],-v0[1])
        if options["isFixNode"] is None:
            self.isFixNode = bm.zeros(node.shape[0], dtype=bool)
        else:
            self.isFixNode = options["isFixNode"]
        self.FixNode = node[self.isFixNode]
        self.isFreeNode = ~self.isFixNode
        self.isBdEdgeNode = options["isBdEdgeNode"]
        self.isBdFaceNode = options["isBdFaceNode"]
        self.isBdNode = self.mesh.boundary_node_flag()
        if self.isFixNode is not None:
            self.isBdNode[self.isFixNode] = False
            x0 = bm.array(node[self.isFreeNode].T.flat)
        else:
            x0 = bm.array(node.T.flat)        
        self.Project = options["Project"] 
        self.Tangent1d = options["Tangent1d"]
        self.Normal2d = options["Normal2d"]
        super().__init__(x0, self.quality)
        self.project_to_boundary = (
            self.project_to_bd if self.Project is not None else None
        )

    @classmethod
    def get_options(
            cls,*,
            mesh:TriangleMesh,
            isFixNode: Optional[TensorLike] = None,
            isBdEdgeNode: Optional[TensorLike] = None,
            isBdFaceNode: Optional[TensorLike] = None,
            Project: Optional[Callable[[TensorLike],TensorLike]] = None,
            Tangent1d: Optional[Callable[[TensorLike, TensorLike], TensorLike]] = None,
            Normal2d: Optional[Callable[[TensorLike], TensorLike]] = None,
            ) -> dict:
        options = {
            'mesh': mesh,
            'isFixNode': isFixNode,
            'isBdEdgeNode': isBdEdgeNode,
            'isBdFaceNode': isBdFaceNode,
            'Project': Project,
            'Tangent1d': Tangent1d,
            'Normal2d': Normal2d,
        }
        return options

    def project_to_bd(self,x=None):
        if x is None:
            x = self.x0
        node = bm.zeros_like(self.mesh.entity('node'))
        node[self.isFixNode] = self.FixNode
        node[self.isFreeNode] = x.reshape(3,-1).T
        nodebd = self.Project(node)
        node[self.isBdFaceNode] = nodebd[self.isBdFaceNode]
        if self.isBdEdgeNode is not None:
            node[self.isBdEdgeNode] = nodebd[self.isBdEdgeNode]
        return node.T.flatten()

    def tangent_project(self,grad,x):
        if grad.ndim==1:
            grad = grad.reshape(3,-1).T
        gradp = bm.full_like(self.mesh.entity('node'),0.0)
        gradp[self.isFreeNode] = grad
        node = bm.full_like(self.mesh.entity('node'),0.0)
        node[self.isFixNode] = self.FixNode
        node[self.isFreeNode] = x.reshape(3,-1).T
        if self.Tangent1d is not None:
            gradp1d = self.Tangent1d(node)
            gradp[self.isBdEdgeNode] = gradp1d[self.isBdEdgeNode]

        if self.Normal2d is not None:
            normal = self.Normal2d(node)
            gradp2d = gradp[self.isBdFaceNode]
            gtd = bm.sum(gradp2d*normal[self.isBdFaceNode],axis=1,keepdims=True)
            gradp2d = gradp2d - gtd*normal[self.isBdFaceNode]
            gradp[self.isBdFaceNode] = gradp2d
        else:
            normal = self.vertex_normal(node)
            gradp2d = gradp[self.isBdFaceNode]
            gtd = bm.sum(gradp2d*normal[self.isBdFaceNode],axis=1,keepdims=True)
            gradp2d = gradp2d - gtd*normal[self.isBdFaceNode]
            gradp[self.isBdFaceNode] = gradp2d
        return gradp[self.isFreeNode]    

    def vertex_normal(self,x):
        if x is None:
            x = self.x0
        node = x.reshape(3,-1).T
        normals = bm.zeros_like(node)
        cell = self.mesh.entity('cell')
        v10 = node[cell[:,1],:] - node[cell[:,0],:]
        v20 = node[cell[:,2],:] - node[cell[:,0],:]
        n = bm.cross(v10,v20)
        for i in range(3):
            bm.index_add(normals, cell[:,i], n)
        norm = bm.linalg.norm(normals,axis=1,keepdims=True)
        norm[norm==0.0] = 1.0
        normals /= norm
        return normals 

    def quality(self,x):
        node0 = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        GD = self.mesh.geo_dimension()
        node = bm.full_like(node0,0.0)
        
        NN = self.mesh.number_of_nodes()
        NC = self.mesh.number_of_cells() 
        
        NI = len(x)//3
        node[:,0] = x[:NI]
        node[:,1] = x[NI:2*NI]
        node[:,2] = x[2*NI:3*NI]

        localEdge = self.mesh.localEdge
        v = [node[cell[:,j],:] - node[cell[:,i],:] for i,j in localEdge]
        l2 = bm.zeros((NC, 3))
        for i in range(3):
            l2[:, i] = bm.sum(v[i]**2, axis=1)
        l = bm.sqrt(l2)
        p = l.sum(axis=1)
        q = l.prod(axis=1)
        nv = bm.cross(v[2],-v[1])
        area = bm.linalg.norm(nv,axis=1)/2

        quality = p*q/(16*area**2)
        penalty = self.penalty(area,nv)
        quality = quality + penalty
        A = self.grad_matrix(node=node)
        gradp = bm.full_like(node,0.0)
        gradp[:,0] = A@node[:,0]
        gradp[:,1] = A@node[:,1]
        gradp[:,2] = A@node[:,2]
        gradp = gradp[~self.isFixNode]
        gradp = self.tangent_project(gradp, x)
        gradp = gradp.T.flatten()
        return bm.mean(quality), gradp

    def penalty(self,q,v):
        pen = bm.zeros_like(q)
        q1 = bm.sum(v*self.nv0,axis=1)
        flag = q1<0
        pen[flag] = (1-q[flag]/1e-12)**2
        return pen

    def grad_matrix(self,node=None):
        NC = self.mesh.number_of_cells()
        NN = self.mesh.number_of_nodes()
        if node is None:
            node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')

        idx0 = cell[:, 0]
        idx1 = cell[:, 1] 
        idx2 = cell[:, 2] 

        v10 = node[idx0] - node[idx1]
        v20 = node[idx0] - node[idx2]
        v12 = node[idx2] - node[idx1]

        area = bm.sqrt(bm.square(bm.cross(v10, v20)).sum(axis=1))/2
        
        l02 = bm.sum(v12**2, axis=-1)
        l12 = bm.sum(v20**2, axis=-1)
        l22 = bm.sum(v10**2, axis=-1)
        l0 = bm.sqrt(l02)
        l1 = bm.sqrt(l12)
        l2 = bm.sqrt(l22)

        p = l0+l1+l2
        q = l0*l1*l2
        mu = p*q/(16*area**2)
        c0 = mu*(1/(p*l0) + 1/(l02))
        c1 = mu*(1/(p*l1) + 1/(l12))
        c2 = mu*(1/(p*l2) + 1/(l22))

        A = bm.zeros((NC,3,3),dtype=self.mesh.ftype)
        S = bm.zeros((NC,3,3),dtype=self.mesh.ftype)
        A[:,0,0] = c1 + c2        
        A[:,1,1] = c0 + c2
        A[:,2,2] = c0 + c1
        A[:,0,1] = -c2
        A[:,0,2] = -c1
        A[:,1,2] = -c0
        A[:,1,0] = A[:,0,1]
        A[:,2,0] = A[:,0,2]
        A[:,2,1] = A[:,1,2]
       
        S[:,0,0] = l02
        S[:,1,1] = l12
        S[:,2,2] = l22
        S[:,0,1] = bm.sum(v20*v12,axis=-1)
        S[:,0,2] = -bm.sum(v10*v12,axis=-1)
        S[:,1,2] = -bm.sum(v10*v20,axis=-1)
        S[:,1,0] = S[:,0,1]
        S[:,2,0] = S[:,0,2]
        S[:,2,1] = S[:,1,2]
        
        t = -mu/(2*area*area)
        S = S*t[:,None,None]
        A+= S
        A/= NC
        I = bm.broadcast_to(cell[:, :, None], (NC, 3, 3))
        J = bm.broadcast_to(cell[:, None, :], (NC, 3, 3))
        indices = bm.stack([I.flatten(),J.flatten()],axis=0)
        A = COOTensor(indices, A.flatten(), spshape=(NN, NN))
        return A

    def grad_matrix_A(self,node=None):
        NC = self.mesh.number_of_cells()
        NN = self.mesh.number_of_nodes()
        if node is None:
            node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')

        idx0 = cell[:, 0]
        idx1 = cell[:, 1] 
        idx2 = cell[:, 2] 

        v10 = node[idx0] - node[idx1]
        v20 = node[idx0] - node[idx2]
        v12 = node[idx2] - node[idx1]

        area = bm.sqrt(bm.square(bm.cross(v10, v20)).sum(axis=1))/2
        
        l02 = bm.sum(v12**2, axis=-1)
        l12 = bm.sum(v20**2, axis=-1)
        l22 = bm.sum(v10**2, axis=-1)
        l0 = bm.sqrt(l02)
        l1 = bm.sqrt(l12)
        l2 = bm.sqrt(l22)

        p = l0+l1+l2
        q = l0*l1*l2
        mu = p*q/(16*area**2)
        c0 = mu*(1/(p*l0) + 1/(l02))
        c1 = mu*(1/(p*l1) + 1/(l12))
        c2 = mu*(1/(p*l2) + 1/(l22))

        A = bm.zeros((NC,3,3),dtype=self.mesh.ftype)
        A[:,0,0] = c1 + c2        
        A[:,1,1] = c0 + c2
        A[:,2,2] = c0 + c1
        A[:,0,1] = -c2
        A[:,0,2] = -c1
        A[:,1,2] = -c0
        A[:,1,0] = A[:,0,1]
        A[:,2,0] = A[:,0,2]
        A[:,2,1] = A[:,1,2]
        A/= NC
        if bm.sum(self.isFixNode)>0:
            isFreeNode = ~self.isFixNode
            NF = bm.sum(isFreeNode)
            old2new = bm.zeros((NN,),dtype=bm.int64)
            old2new[isFreeNode] = bm.arange(NF,dtype=bm.int64)
            I_flat = I.flatten()
            J_flat = J.flatten()
            V_flat = A.flatten()
            keep = isFreeNode[I_flat] & isFreeNode[J_flat]
            I_flat = old2new[I_flat[keep]]
            J_flat = old2new[J_flat[keep]]
            V_flat = V_flat[keep]
            indices = bm.stack([I_flat,J_flat],axis=0)
            A = COOTensor(indices,V_flat,spshape=(NF, NF))
            return A
        I = bm.broadcast_to(cell[:, :, None], (NC, 3, 3))
        J = bm.broadcast_to(cell[:, None, :], (NC, 3, 3))
        indice = bm.stack([I.flatten(),J.flatten()],axis=0)
        A = COOTensor(indice, A.flatten(), spshape=(NN, NN))
        return A

    def preconditioner_tangent(self,node=None):
        if node is None:
            node = self.mesh.entity('node').copy()
        NN = node.shape[0]
        isBdNode = self.isBdNode
        isBdFaceNode = self.isBdFaceNode
        Pi = bm.zeros((NN,3,3),dtype=self.mesh.ftype)
        Pi[~isBdNode] = bm.eye(3)
        grad = bm.ones_like(node,dtype=self.mesh.ftype)
        if self.Tangent1d is not None:
            grad1d = self.Tangent1d(grad,node)
            grad1d = grad1d[self.isBdEdgeNode]
            grad1d = grad1d/bm.linalg.norm(grad1d,axis=1,keepdims=True)
            Pi[self.isBdEdgeNode] = bm.einsum('ij,ik->ijk',grad1d,grad1d)
        if self.Normal2d is not None:
            normal = self.Normal2d(node)
            snormal = normal[self.isBdFaceNode]
            snormal = snormal/bm.linalg.norm(snormal,axis=1,keepdims=True)
            Pi[self.isBdFaceNode] = bm.eye(3) -bm.einsum('ij,ik->ijk',snormal,snormal)
        else:
            normal = self.vertex_normal(node)
            snormal = normal[isBdFaceNode]
            snormal = snormal/bm.linalg.norm(snormal,axis=1,keepdims=True)
            Pi[self.isBdFaceNode] = bm.eye(3) -bm.einsum('ij,ik->ijk',snormal,snormal)
        
        if bm.sum(self.isFixNode)>0:
            isFreeNode = ~self.isFixNode
            NF = bm.sum(isFreeNode)
            Pi = Pi[isFreeNode]
            ids = bm.arange(NF,dtype=bm.int64)
            dof = bm.stack([ids,ids+NF,ids+2*NF],axis=1)
            rows = bm.repeat(dof,3,axis=1)
            cols = bm.tile(dof,(1,3))
            indices = bm.stack([rows.flatten(), cols.flatten()],axis=0)
            values = Pi.flatten()
            Pi = COOTensor(indices,values,spshape=(3*NF, 3*NF))
            return Pi
        ids = bm.arange(NN,dtype=bm.int64)
        dof = bm.stack([ids,ids+NN,ids+2*NN],axis=1)
        rows = bm.repeat(dof,3,axis=1)
        cols = bm.tile(dof,(1,3))
        indices = bm.stack([rows.flatten(),cols.flatten()],axis=0)
        Pi = COOTensor(indices, Pi.flatten(), spshape=(3*NN, 3*NN))
        return Pi

    def build_preconditioner_matrices(self,x=None):
        if x is None:
            x = self.x0
        isFreeNode = ~self.isFixNode
        isFreeNode3 = bm.repeat(isFreeNode,3)
        node0 = self.mesh.entity('node').copy()
        n = len(x)//3
        node0[isFreeNode,0] = x[:n]
        node0[isFreeNode,1] = x[n:2*n]
        node0[isFreeNode,2] = x[2*n:]
        A = self.grad_matrix_A(node0)
        Pi = self.preconditioner_tangent(node0)
        if self.Tangent1d is None:
            nA = A.sparse_shape[0]
            idx = bm.arange(nA,dtype=bm.int64)
            indices = bm.stack([idx,idx],axis=0)
            values = bm.full((nA,),1e-4,dtype=self.mesh.ftype)
            I = COOTensor(indices,values,spshape=(nA,nA))
            A = A.add(I)

        nA = A.sparse_shape[0]
        idx0 = A.indices
        val0 = A.values
        idx1 = bm.copy(idx0)
        idx1[0] = idx1[0]+nA
        idx1[1] = idx1[1]+nA
        idx2 = bm.copy(idx0)
        idx2[0] = idx2[0]+2*nA
        idx2[1] = idx2[1]+2*nA
        indices = bm.concatenate([idx0,idx1,idx2],axis=1)
        values = bm.concatenate([val0,val0,val0],axis=0)
        D = COOTensor(indices,values,spshape=(3*nA,3*nA))
        return D, Pi
    def build_preconditioner(self, update_interval=3, rtol=1e-2, maxiter=100):
        return ProjectedCGPreconditioner(
            self,
            update_interval=update_interval,
            rtol=rtol,
            maxiter=maxiter,
        )
    @staticmethod
    def get_quality(mesh):
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        GD = mesh.geo_dimension()
        
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells() 
        
        localEdge = mesh.localEdge
        v = [node[cell[:,j],:] - node[cell[:,i],:] for i,j in localEdge]
        l2 = bm.zeros((NC, 3))
        for i in range(3):
            l2[:, i] = bm.sum(v[i]**2, axis=1)
        l = bm.sqrt(l2)
        p = l.sum(axis=1)
        q = l.prod(axis=1)
        nv = bm.cross(v[2],-v[1])
        area = bm.linalg.norm(nv,axis=1)/2

        quality = p*q/(16*area**2)
        return quality

