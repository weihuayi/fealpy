from typing import TypedDict, Callable, Tuple, Union, Optional

from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike 
from fealpy.sparse import COOTensor,CSRTensor
from fealpy.mesh import TriangleMesh

from opt import Problem
from .rropreconditioner import ProjectedCGPreconditioner

class TriMeshProblem(Problem):
    def __init__(self, options: dict):
        self.mesh = options["mesh"]
        self.FixAllBoundary = options["FixAllBoundary"]
        node = self.mesh.entity('node')
        if options["isFixNode"] is None:
            self.isFixNode = bm.zeros(node.shape[0], dtype=bm.bool)
        else:
            self.isFixNode = options["isFixNode"]
        self.isFreeNode = ~self.isFixNode
        self.isBdEdgeNode = options["isBdEdgeNode"]
        self.isBdNode = self.mesh.boundary_node_flag()
        if self.FixAllBoundary is True:
            self.isFixNode[self.isBdNode] = True
            self.isFreeNode[self.isBdNode] = False
            x0 = bm.array(node[self.isFreeNode].T.flat)
        elif self.isFixNode is not None:
            self.isBdNode[self.isFixNode] = False
            x0 = bm.array(node[self.isFreeNode].T.flat) 
        else:
            x0 = bm.array(node.T.flat)
        self.FixNode = node[self.isFixNode]
        self.Project = options["Project"] 
        self.Tangent1d = options["Tangent1d"]
        super().__init__(x0,self.quality)
        self.quality(x0)
        self.project_to_boundary = (
                self.project_to_bd if self.Project is not None else None
                )

    @classmethod
    def get_options(
        cls,*,
        mesh:TriangleMesh,
        FixAllBoundary: bool = False,
        isFixNode: Optional[TensorLike] = None,
        isBdEdgeNode: Optional[TensorLike] = None,
        Project: Optional[Callable[[TensorLike],TensorLike]] = None,
        Tangent1d: Optional[Callable[[TensorLike, TensorLike], TensorLike]] = None,
        ) -> dict:
        options = {
            'mesh': mesh,
            'FixAllBoundary': FixAllBoundary,
            'isFixNode': isFixNode,
            'isBdEdgeNode': isBdEdgeNode,
            'Project': Project,
            'Tangent1d': Tangent1d,
        }
        return options

    def project_to_bd(self,x=None):
        if x is None:
            x = self.x0
        node = bm.zeros_like(self.mesh.entity('node'))
        node[self.isFixNode] = self.FixNode
        node[self.isFreeNode] = x.reshape(2,-1).T
        nodebd = self.Project(node)
        node[self.isBdNode] = nodebd[self.isBdNode]
        return node[self.isFreeNode].T.flatten()
    
    def tangent_project(self,grad,x):
        if grad.ndim==1:
            grad = grad.reshape(2,-1).T
        gradp = bm.full_like(self.mesh.entity('node'),0.0)
        gradp[self.isFreeNode] = grad
        node = bm.full_like(self.mesh.entity('node'),0.0)
        node[self.isFixNode] = self.FixNode
        node[self.isFreeNode] = x.reshape(2,-1).T
        gradp1d = self.Tangent1d(gradp,node)
        gradp[self.isBdEdgeNode] = gradp1d[self.isBdEdgeNode]
        return gradp[self.isFreeNode]

    def quality(self,x):
        node0 = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        NC = self.mesh.number_of_cells()
        NN = self.mesh.number_of_nodes()

        node = bm.zeros_like(node0)
        node[self.isFixNode] = self.FixNode
        n = len(x)//2
        node[self.isFreeNode,0] = x[:n]
        node[self.isFreeNode,1] = x[n:]

        localEdge = self.mesh.localEdge
        idxi = cell[:, 0]
        idxj = cell[:, 1] 
        idxk = cell[:, 2]
        v0 = node[idxk] - node[idxj]
        v1 = node[idxi] - node[idxk]
        v2 = node[idxj] - node[idxi]

        #area = 0.5*(-v2[:, 0]*v1[:, 1] + v2[:, 1]*v1[:, 0])
        area = bm.cross(v2,-v1)/2
        area = area[:,None] 
        l2 = bm.zeros((NC, 3), dtype=bm.float64)
        l2[:, 0] = bm.sum(v0**2, axis=1)
        l2[:, 1] = bm.sum(v1**2, axis=1)
        l2[:, 2] = bm.sum(v2**2, axis=1) 
        l = bm.sqrt(l2)

        p = l.sum(axis=1,keepdims=True)
        q = l.prod(axis=1,keepdims=True)
        mu = p*q/(16*area**2)
        penalty = self.penalty(area)
        mu = mu+penalty
        c = mu*(1/(p*l) + 1/l2)
        val = bm.concatenate((
            c[:, [1, 2]].sum(axis=1), -c[:, 2], -c[:, 1],
            -c[:, 2], c[:, [0, 2]].sum(axis=1), -c[:, 0],
            -c[:, 1], -c[:, 0], c[:, [0, 1]].sum(axis=1)))
        I = bm.concatenate((
            idxi, idxi, idxi,
            idxj, idxj, idxj,
            idxk, idxk, idxk))
        J = bm.concatenate((idxi, idxj, idxk))
        J = bm.concatenate((J, J, J))
        indice = bm.stack([I,J],axis=0)
        #A = csr_matrix((val, (I, J)), shape=(NN, NN))
        A = COOTensor(indice, val,spshape=(NN, NN))
        A = A/NC
        self.A = A
        cn = mu/area
        cn.shape = (cn.shape[0],)
        val = bm.concatenate((-cn, cn, cn, -cn, -cn, cn))
        I = bm.concatenate((idxi, idxi, idxj, idxj, idxk, idxk))
        J = bm.concatenate((idxj, idxk, idxi, idxk, idxi, idxj)) 
        indice = bm.stack([I,J],axis=0)
        #B = csr_matrix((val, (I, J)), shape=(NN, NN))
        B = COOTensor(indice,val, spshape=(NN, NN))
        B = B/NC
        gradp = bm.full_like(node,0.0)
        if self.FixAllBoundary is True:
            gradp[:,0] = A@node[:,0]+B@node[:,1]
            gradp[:,1] = B.T@node[:,0]+A@node[:,1]
            gradp = gradp[~self.isFixNode]
            gradp = gradp.T.flatten()
            return bm.mean(mu),gradp
        gradp[:,0] = A@node[:,0]+B@node[:,1]
        gradp[:,1] = B.T@node[:,0]+A@node[:,1]
        gradp = gradp[~self.isFixNode]
        gradp = self.tangent_project(gradp,x)
        gradp = gradp.T.flatten()
        return bm.mean(mu),gradp
    def penalty(self,q):
        pen = bm.zeros_like(q)
        flag = q<0
        pen[flag] = (1-q[flag]/1e-12)**2
        return pen

    def preconditioner_tangent(self,node=None):
        if node is None:
            node = self.mesh.entity('node').copy()
        NN = node.shape[0]
        isBdNode = self.isBdNode
        Pi = bm.zeros((NN,2,2),dtype=self.mesh.ftype)
        Pi[~isBdNode] = bm.eye(2)
        if self.FixAllBoundary is True:
            Pi = Pi[~isBdNode]
            NF = bm.sum(~isBdNode)
            ids = bm.arange(NF,dtype=bm.int64)
            dof = bm.stack([ids,ids+NF],axis=1)
            rows = bm.repeat(dof,2,axis=1)
            cols = bm.tile(dof,(1,2))
            indices = bm.stack([rows.flatten(), cols.flatten()],axis=0)
            #Pi = csr_matrix((Pi.flat, (rows.flat, cols.flat)), shape=(3*NN, 3*NN))
            values = Pi.flatten()
            Pi = COOTensor(indices,values,spshape=(2*NF, 2*NF))
            return Pi
        grad = bm.ones_like(node,dtype=self.mesh.ftype)
        grad1d = self.Tangent1d(grad,node)
        grad1d = grad1d[self.isBdEdgeNode]
        grad1d = grad1d/bm.linalg.norm(grad1d,axis=1,keepdims=True)
        Pi[self.isBdEdgeNode] = bm.einsum('ij, ik->ijk', grad1d, grad1d)
        if bm.sum(self.isFixNode)>0:
            isFreeNode = ~self.isFixNode
            NF = bm.sum(isFreeNode)
            Pi = Pi[isFreeNode]
            ids = bm.arange(NF,dtype=bm.int64)
            dof = bm.stack([ids,ids+NF],axis=1)
            rows = bm.repeat(dof,2,axis=1)
            cols = bm.tile(dof, (1,2))
            indices = bm.stack([rows.flatten(), cols.flatten()],axis=0)
            values = Pi.flatten()
            Pi = COOTensor(indices, values, spshape=(2*NF, 2*NF))
            return Pi
        ids = bm.arange(NN,dtype=bm.int64)
        dof = bm.stack([ids,ids+NN],axis=1)
        rows = bm.repeat(dof,2,axis=1)
        cols = bm.tile(dof, (1,2))
        indices = bm.stack([rows.flatten(), cols.flatten()],axis=0)
        #Pi = csr_matrix((Pi.flat,(rows.flat,cols.flat)),shape=(2*NN, 2*NN))
        Pi = COOTensor(indices,Pi.flatten(),spshape=(2*NN, 2*NN))
        return Pi

    def build_preconditioner_matrices(self,x=None):
        if x is None:
            x = self.x0
        isFreeNode = ~self.isFixNode
        isFreeNode2 = bm.repeat(isFreeNode,2)
        node0 = self.mesh.entity('node').copy()
        n = len(x)//2
        node0[isFreeNode,0] = x[:n]
        node0[isFreeNode,1] = x[n:]
        #A = self.A
        A = self.grad_matrix_A(node0)
        Pi = self.preconditioner_tangent(node0)

        nA = A.sparse_shape[0]
        idx0 = A.indices
        val0 = A.values
        idx1 = bm.copy(idx0)
        idx1[0] = idx1[0]+nA
        idx1[1] = idx1[1]+nA
        indices = bm.concatenate([idx0,idx1],axis=1)
        values = bm.concatenate([val0,val0],axis=0)
        D = COOTensor(indices,values,spshape=(2*nA,2*nA))
        return D,Pi

    def build_preconditioner(self, update_interval=3, rtol=1e-2, maxiter=100):
        return ProjectedCGPreconditioner(
            self,
            update_interval=update_interval,
            rtol=rtol,
            maxiter=maxiter,
        )

    def grad_matrix_A(self,node=None):
        if node is None:
            node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        NC = self.mesh.number_of_cells()
        NN = self.mesh.number_of_nodes()

        localEdge = self.mesh.localEdge
        idxi = cell[:, 0]
        idxj = cell[:, 1] 
        idxk = cell[:, 2]
        v0 = node[idxk] - node[idxj]
        v1 = node[idxi] - node[idxk]
        v2 = node[idxj] - node[idxi]

        area = bm.cross(v2,-v1)/2
        area = area[:,None] 
        l2 = bm.zeros((NC, 3), dtype=bm.float64)
        l2[:, 0] = bm.sum(v0**2, axis=1)
        l2[:, 1] = bm.sum(v1**2, axis=1)
        l2[:, 2] = bm.sum(v2**2, axis=1) 
        l = bm.sqrt(l2)

        p = l.sum(axis=1,keepdims=True)
        q = l.prod(axis=1,keepdims=True)
        mu = p*q/(16*area**2)
        c = mu*(1/(p*l) + 1/l2)
        val = bm.concatenate((
            c[:, [1, 2]].sum(axis=1), -c[:, 2], -c[:, 1],
            -c[:, 2], c[:, [0, 2]].sum(axis=1), -c[:, 0],
            -c[:, 1], -c[:, 0], c[:, [0, 1]].sum(axis=1)))
        I = bm.concatenate((
            idxi, idxi, idxi,
            idxj, idxj, idxj,
            idxk, idxk, idxk))
        J = bm.concatenate((idxi, idxj, idxk))
        J = bm.concatenate((J, J, J))
        if bm.sum(self.isFixNode)>0:
            isFreeNode = ~self.isFixNode
            NF = bm.sum(isFreeNode) 
            old2new = bm.zeros((NN,), dtype=bm.int64)
            old2new[isFreeNode] = bm.arange(NF,dtype=bm.int64)
            keep = isFreeNode[I] & isFreeNode[J]
            I = old2new[I[keep]]
            J = old2new[J[keep]]
            val = val[keep]
            indices = bm.stack([I,J],axis=0)
            A = COOTensor(indices, val, spshape=(NF,NF))
            A = A/NC
            return A
        indice = bm.stack([I,J],axis=0)
        #A = csr_matrix((val, (I, J)), shape=(NN, NN))
        A = COOTensor(indice, val,spshape=(NN, NN))
        A = A/NC
        return A

    @staticmethod
    def get_quality(mesh):
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()

        localEdge = mesh.localEdge
        idxi = cell[:, 0]
        idxj = cell[:, 1] 
        idxk = cell[:, 2]
        v0 = node[idxk] - node[idxj]
        v1 = node[idxi] - node[idxk]
        v2 = node[idxj] - node[idxi]

        area = bm.cross(v2,-v1)/2
        area = area[:,None] 
        l2 = bm.zeros((NC, 3), dtype=bm.float64)
        l2[:, 0] = bm.sum(v0**2, axis=1)
        l2[:, 1] = bm.sum(v1**2, axis=1)
        l2[:, 2] = bm.sum(v2**2, axis=1) 
        l = bm.sqrt(l2)

        p = l.sum(axis=1,keepdims=True)
        q = l.prod(axis=1,keepdims=True)
        quality = p*q/(16*area**2)
        return quality 
