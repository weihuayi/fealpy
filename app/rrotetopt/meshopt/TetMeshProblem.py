from typing import TypedDict, Callable, Tuple, Union, Optional

from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike 
from fealpy.mesh import TetrahedronMesh
from fealpy.sparse import COOTensor,CSRTensor

from opt import Problem
from .rropreconditioner import ProjectedCGPreconditioner

import matplotlib.pyplot as plt

class TetMeshProblem(Problem):
    def __init__(self,options: dict):
        self.mesh = options["mesh"]
        self.FixAllBoundary = options["FixAllBoundary"]
        node = self.mesh.entity('node')
        if options["isFixNode"] is None:
            self.isFixNode = bm.zeros(node.shape[0],dtype=bool)
        else:
            self.isFixNode = options["isFixNode"]
        self.isFreeNode = ~self.isFixNode
        self.isBdEdgeNode = options["isBdEdgeNode"]
        self.isBdFaceNode = options["isBdFaceNode"]
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
        self.Normal2d = options["Normal2d"]
        self._NFlip = 0
        super().__init__(x0, self.quality)
        self.project_to_boundary = (
            self.project_to_bd if self.Project is not None else None
        )

    @classmethod
    def get_options(
            cls,*,
            mesh:TetrahedronMesh,
            FixAllBoundary: bool = False,
            isFixNode: Optional[TensorLike] = None,
            isBdEdgeNode: Optional[TensorLike] = None,
            isBdFaceNode: Optional[TensorLike] = None,
            Project: Optional[Callable[[TensorLike],TensorLike]] = None,
            Tangent1d: Optional[Callable[[TensorLike, TensorLike], TensorLike]] = None,
            Normal2d: Optional[Callable[[TensorLike], TensorLike]] = None
            ) -> dict:
        options = {
            'mesh': mesh,
            'FixAllBoundary': FixAllBoundary,
            'isFixNode': isFixNode,
            'isBdEdgeNode': isBdEdgeNode,
            'isBdFaceNode': isBdFaceNode,
            'Project': Project,
            'Tangent1d': Tangent1d,
            'Normal2d': Normal2d
        }
        return options

    def project_to_bd(self,x=None):
        if x is None:
            x = self.x0
        node = bm.zeros_like(self.mesh.entity('node'))
        node[self.isFixNode] = self.FixNode
        node[self.isFreeNode] = x.reshape(3,-1).T
        nodebd = self.Project(node)
        node[self.isBdNode] = nodebd[self.isBdNode]
        return node[self.isFreeNode].T.flatten()

    def tangent_project(self,grad,x):
        if grad.ndim==1:
            grad = grad.reshape(3,-1).T
        gradp = bm.full_like(self.mesh.entity('node'),0.0)
        gradp[self.isFreeNode] = grad
        node = bm.full_like(self.mesh.entity('node'),0.0)
        node[self.isFixNode] = self.FixNode
        node[self.isFreeNode] = x.reshape(3,-1).T
        if self.Tangent1d is not None:
            gradp1d = self.Tangent1d(gradp,node)
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

    def vertex_normal(self,node):
        if node is None:
            node = self.mesh.entity('node')
        face = self.mesh.entity('face')
        boundary_face_flag = self.mesh.boundary_face_flag()
        boundary_node_flag = self.mesh.boundary_node_flag()
        bdface = face[boundary_face_flag]
        normals = bm.zeros_like(node)
        v10 = node[bdface[:,1],:] - node[bdface[:,0],:]
        v20 = node[bdface[:,2],:] - node[bdface[:,0],:]
        n = bm.cross(v10,v20)
        for i in range(3):
            bm.index_add(normals, bdface[:,i], n)
        norm = bm.linalg.norm(normals,axis=1,keepdims=True)
        norm[norm==0.0] = 1.0
        normals /= norm
        return normals

    def quality(self, x):
        GD = self.mesh.geo_dimension()
        node0 = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        node = bm.zeros_like(node0)
        node[self.isFixNode] = self.FixNode
        NN = self.mesh.number_of_nodes()
        NC = self.mesh.number_of_cells()
        
        n = len(x)//3
        node[self.isFreeNode,0] = x[:n]
        node[self.isFreeNode,1] = x[n:2*n]
        node[self.isFreeNode,2] = x[2*n:]

        v10 = node[cell[:, 0]] - node[cell[:, 1]]
        v20 = node[cell[:, 0]] - node[cell[:, 2]]
        v30 = node[cell[:, 0]] - node[cell[:, 3]]
        v21 = node[cell[:, 1]] - node[cell[:, 2]]
        v31 = node[cell[:, 1]] - node[cell[:, 3]]
        v32 = node[cell[:, 2]] - node[cell[:, 3]]

        l10 = bm.sum(v10**2, axis=-1)
        l20 = bm.sum(v20**2, axis=-1)
        l30 = bm.sum(v30**2, axis=-1)
        l21 = bm.sum(v21**2, axis=-1)
        l31 = bm.sum(v31**2, axis=-1)
        l32 = bm.sum(v32**2, axis=-1)

        d0 = bm.zeros((NC, 3), dtype=self.mesh.ftype)
        c12 = bm.cross(v10, v20)
        d0 += l30[:, None]*c12
        c23 = bm.cross(v20, v30)
        d0 += l10[:, None]*c23
        c31 = bm.cross(v30, v10)
        d0 += l20[:, None]*c31

        c12 = bm.sum(c12*d0, axis=-1)
        c23 = bm.sum(c23*d0, axis=-1)
        c31 = bm.sum(c31*d0, axis=-1)
        c = c12 + c23 + c31
        ld0 = bm.sum(d0**2, axis=-1)

        face = self.mesh.entity('face')
        fv01 = node[face[:, 1], :] - node[face[:, 0], :]
        fv02 = node[face[:, 2], :] - node[face[:, 0], :]
        fm = bm.sqrt(bm.square(bm.cross(fv01,fv02)).sum(axis=1))/2.0

        cm = bm.sum(-v30*bm.cross(v10,v20),axis=1)/6.0
        c2f = self.mesh.cell_to_face()
        
        s = fm[c2f]
        s_sum = bm.sum(s, axis=-1)
        quality = s_sum*bm.sqrt(ld0)/(108*cm*cm)
        penalty = self.penalty(cm)
        quality = quality+penalty
        A = bm.zeros((NC, 4, 4), dtype=self.mesh.ftype)
        A[:, 0, 0]  = 2*c
        A[:, 0, 1] -= 2*c23
        A[:, 0, 2] -= 2*c31
        A[:, 0, 3] -= 2*c12

        A[:, 1, 1] = 2*c23
        A[:, 2, 2] = 2*c31
        A[:, 3, 3] = 2*c12
        A[:, 1:, 0] = A[:, 0, 1:]        

        K = bm.zeros((NC, 4, 4), dtype=self.mesh.ftype)
        K[:, 0, 1] -= l30 - l20
        K[:, 0, 2] -= l10 - l30
        K[:, 0, 3] -= l20 - l10
        K[:, 1:, 0] -= K[:, 0, 1:]

        K[:, 1, 2] -= l30
        K[:, 1, 3] += l20
        K[:, 2:, 1] -= K[:, 1, 2:]

        K[:, 2, 3] -= l10
        K[:, 3, 2] += l10

        p0 = (l31/s[:,2] + l21/s[:,3] + l32/s[:,1])/4
        p1 = (l32/s[:,0] + l20/s[:,3] + l30/s[:,2])/4
        p2 = (l30/s[:,1] + l10/s[:,3] + l31/s[:,0])/4
        p3 = (l10/s[:,2] + l20/s[:,1] + l21/s[:,0])/4

        q10 = -(bm.sum(v31*v30, axis=-1)/s[:,2]+bm.sum(v21*v20, axis=-1)/s[:,3])/4
        q20 = -(bm.sum(v32*v30, axis=-1)/s[:,1]+bm.sum(-v21*v10, axis=-1)/s[:,3])/4
        q30 = -(bm.sum(-v32*v20, axis=-1)/s[:,1]+bm.sum(-v31*v10, axis=-1)/s[:,2])/4
        q21 = -(bm.sum(v32*v31, axis=-1)/s[:,0]+bm.sum(v20*v10, axis=-1)/s[:,3])/4
        q31 = -(bm.sum(v30*v10, axis=-1)/s[:,2]+bm.sum(-v32*v21, axis=-1)/s[:,0])/4
        q32 = -(bm.sum(v31*v21, axis=-1)/s[:,0]+bm.sum(v30*v20, axis=-1)/s[:,1])/4
        
        S = bm.zeros((NC, 4, 4), dtype=self.mesh.ftype)
        S[:, 0, 0] = p0
        S[:, 0, 1] = q10
        S[:, 0, 2] = q20
        S[:, 0, 3] = q30
        S[:, 1:,0] = S[:, 0, 1:]

        S[:, 1, 1] = p1
        S[:, 1, 2] = q21
        S[:, 1, 3] = q31
        S[:, 2:,1] = S[:, 1, 2:]

        S[:, 2, 2] = p2
        S[:, 2, 3] = q32
        S[:, 3, 2] = q32
        S[:, 3, 3] = p3

        C0 = bm.zeros((NC, 4, 4), dtype=bm.float64)
        C1 = bm.zeros((NC, 4, 4), dtype=bm.float64)
        C2 = bm.zeros((NC, 4, 4), dtype=bm.float64)

        def f(CC, xx):
            CC[:, 0, 1] = xx[:, 2]
            CC[:, 0, 2] = xx[:, 3]
            CC[:, 0, 3] = xx[:, 1]
            CC[:, 1, 0] = xx[:, 3]
            CC[:, 1, 2] = xx[:, 0]
            CC[:, 1, 3] = xx[:, 2]
            CC[:, 2, 0] = xx[:, 1]
            CC[:, 2, 1] = xx[:, 3]
            CC[:, 2, 3] = xx[:, 0]
            CC[:, 3, 0] = xx[:, 2]
            CC[:, 3, 1] = xx[:, 0]
            CC[:, 3, 2] = xx[:, 1]

        f(C0, node[cell, 0])
        f(C1, node[cell, 1])
        f(C2, node[cell, 2])

        C0 = 0.5*(-C0 + C0.swapaxes(-1, -2))
        C1 = 0.5*(C1  - C1.swapaxes(-1, -2))
        C2 = 0.5*(-C2 + C2.swapaxes(-1, -2))

        B0 = -d0[:,0,None,None]*K
        B1 = d0[:,1,None,None]*K
        B2 = -d0[:,2,None,None]*K

        ld0 = bm.sum(d0**2,axis=-1)

        A  /= ld0[:,None,None]
        B0 /= ld0[:,None,None]
        B1 /= ld0[:,None,None]
        B2 /= ld0[:,None,None]

        S  /= s_sum[:,None,None]

        C0 /= 3*cm[:,None,None]
        C1 /= 3*cm[:,None,None]
        C2 /= 3*cm[:,None,None]

        A  += S
        B0 -= C0
        B1 -= C1
        B2 -= C2

        mu = s_sum*bm.sqrt(ld0)/(108*cm**2)

        A *= mu[:,None,None]/NC
        B0 *= mu[:,None,None]/NC
        B1 *= mu[:,None,None]/NC
        B2 *= mu[:,None,None]/NC

        I = bm.broadcast_to(cell[:, :, None], (NC, 4, 4))
        J = bm.broadcast_to(cell[:, None, :], (NC, 4, 4))
        indice = bm.stack([I.flatten(),J.flatten()],axis=0)

        A = COOTensor(indice,A.flatten(),spshape=(NN, NN))
        B0 = COOTensor(indice,B0.flatten(),spshape=(NN, NN))
        B1 = COOTensor(indice,B1.flatten(),spshape=(NN, NN))
        B2 = COOTensor(indice,B2.flatten(),spshape=(NN, NN))

        gradp = bm.full_like(node,0.0)
        if self.FixAllBoundary is True:
            gradp[:,0] = A@node[:,0]+B2@node[:,1]+B1@node[:,2]
            gradp[:,1] = B2.T@node[:,0]+A@node[:,1]+B0@node[:,2]
            gradp[:,2] = B1.T@node[:,0]+B0.T@node[:,1]+A@node[:,2]
            gradp = gradp[~self.isFixNode]
            gradp = gradp.T.flatten()
            return bm.mean(quality), gradp
        gradp[:,0] = (A@node[:,0]+B2@node[:,1]+B1@node[:,2])
        gradp[:,1] = (B2.T@node[:,0]+A@node[:,1]+B0@node[:,2])
        gradp[:,2] = (B1.T@node[:,0]+B0.T@node[:,1]+A@node[:,2])
        gradp = gradp[~self.isFixNode]
        gradp = self.tangent_project(gradp,x)
        gradp = gradp.T.flatten()
        return bm.mean(quality), gradp

    def penalty(self,q):
        pen = bm.zeros_like(q)
        flag = q<0
        pen[flag] = (1-q[flag]/1e-12)**2
        return pen

    def grad_matrix_A(self, node=None):
        NC = self.mesh.number_of_cells()
        NN = self.mesh.number_of_nodes()
        if node is None:
            node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        v10 = node[cell[:, 0]] - node[cell[:, 1]]
        v20 = node[cell[:, 0]] - node[cell[:, 2]]
        v30 = node[cell[:, 0]] - node[cell[:, 3]]

        v21 = node[cell[:, 1]] - node[cell[:, 2]]
        v31 = node[cell[:, 1]] - node[cell[:, 3]]
        v32 = node[cell[:, 2]] - node[cell[:, 3]]

        l10 = bm.sum(v10**2, axis=-1)
        l20 = bm.sum(v20**2, axis=-1)
        l30 = bm.sum(v30**2, axis=-1)
        l21 = bm.sum(v21**2, axis=-1)
        l31 = bm.sum(v31**2, axis=-1)
        l32 = bm.sum(v32**2, axis=-1)

        d0 = bm.zeros((NC, 3), dtype=self.mesh.ftype)
        c12 = bm.cross(v10, v20)
        d0 += l30[:, None]*c12
        c23 = bm.cross(v20, v30)
        d0 += l10[:, None]*c23
        c31 = bm.cross(v30, v10)
        d0 += l20[:, None]*c31

        c12 = bm.sum(c12*d0, axis=-1)
        c23 = bm.sum(c23*d0, axis=-1)
        c31 = bm.sum(c31*d0, axis=-1)
        c12 = bm.abs(c12)
        c23 = bm.abs(c23)
        c31 = bm.abs(c31)
        c = c12 + c23 + c31

        A = bm.zeros((NC, 4, 4), dtype=self.mesh.ftype)
        A[:, 0, 0]  = 2*c
        A[:, 0, 1] -= 2*c23
        A[:, 0, 2] -= 2*c31
        A[:, 0, 3] -= 2*c12

        A[:, 1, 1] = 2*c23
        A[:, 2, 2] = 2*c31
        A[:, 3, 3] = 2*c12
        A[:, 1:, 0] = A[:, 0, 1:]

        S = bm.zeros((NC, 4, 4), dtype=self.mesh.ftype)
        face = self.mesh.entity('face')
        fv01 = node[face[:, 1], :] - node[face[:, 0], :]
        fv02 = node[face[:, 2], :] - node[face[:, 0], :]
        fm = bm.sqrt(bm.square(bm.cross(fv01,fv02)).sum(axis=1))/2.0

        cm = bm.sum(-v30*bm.cross(v10,v20),axis=1)/6.0
        c2f = self.mesh.cell_to_face()

        s = fm[c2f]
        s_sum = bm.sum(s, axis=-1)

        p0 = (l31/s[:,2] + l21/s[:,3] + l32/s[:,1])/4
        p1 = (l32/s[:,0] + l20/s[:,3] + l30/s[:,2])/4
        p2 = (l30/s[:,1] + l10/s[:,3] + l31/s[:,0])/4
        p3 = (l10/s[:,2] + l20/s[:,1] + l21/s[:,0])/4

        q10 = -(bm.sum(v31*v30, axis=-1)/s[:,2]+bm.sum(v21*v20, axis=-1)/s[:,3])/4
        q20 = -(bm.sum(v32*v30, axis=-1)/s[:,1]+bm.sum(-v21*v10, axis=-1)/s[:,3])/4
        q30 = -(bm.sum(-v32*v20, axis=-1)/s[:,1]+bm.sum(-v31*v10, axis=-1)/s[:,2])/4
        q21 = -(bm.sum(v32*v31, axis=-1)/s[:,0]+bm.sum(v20*v10, axis=-1)/s[:,3])/4
        q31 = -(bm.sum(v30*v10, axis=-1)/s[:,2]+bm.sum(-v32*v21, axis=-1)/s[:,0])/4
        q32 = -(bm.sum(v31*v21, axis=-1)/s[:,0]+bm.sum(v30*v20, axis=-1)/s[:,1])/4

        S[:, 0, 0] = p0
        S[:, 0, 1] = q10
        S[:, 0, 2] = q20
        S[:, 0, 3] = q30
        S[:, 1:,0] = S[:, 0, 1:]

        S[:, 1, 1] = p1
        S[:, 1, 2] = q21
        S[:, 1, 3] = q31
        S[:, 2:,1] = S[:, 1, 2:]

        S[:, 2, 2] = p2
        S[:, 2, 3] = q32
        S[:, 3, 2] = q32
        S[:, 3, 3] = p3

        ld0 = bm.sum(d0**2,axis=-1)

        A  /= ld0[:,None,None]
        S  /= s_sum[:,None,None]

        A  += S

        mu = s_sum*bm.sqrt(ld0)/(108*cm**2)

        A  *= mu[:,None,None]/NC
        I = bm.broadcast_to(cell[:, :, None], (NC, 4, 4))
        J = bm.broadcast_to(cell[:, None, :], (NC, 4, 4)) 
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
        indice = bm.stack([I.flatten(),J.flatten()],axis=0)
        A = COOTensor(indice,A.flatten(),spshape=(NN, NN))
        return A

    def preconditioner_tangent(self,node=None):
        if node is None:
            node = self.mesh.entity('node').copy()
        NN = node.shape[0]
        isBdNode = self.isBdNode
        isBdFaceNode = self.isBdFaceNode
        Pi = bm.zeros((NN,3,3),dtype=self.mesh.ftype)
        Pi[~isBdNode] = bm.eye(3)
        if self.FixAllBoundary is True:
            Pi[isBdNode] = bm.eye(3)
            ids = bm.arange(NN,dtype=bm.int64)
            dof = bm.stack([ids,ids+NN,ids+2*NN],axis=1)
            rows = bm.repeat(dof,3,axis=1)
            cols = bm.tile(dof,(1,3))
            indices = bm.stack([rows.flat, cols.flat],axis=0)
            Pi = COOTensor(indices,Pi.flatten(),spshape=(3*NN, 3*NN))
            return Pi
        grad = bm.ones_like(node,dtype=self.mesh.ftype)
        if self.Tangent1d is not None:
            grad1d = self.Tangent1d(grad,node)
            grad1d = grad1d[self.isBdEdgeNode]
            grad1d = grad1d/bm.linalg.norm(grad1d,axis=1,keepdims=True)
            Pi[self.isBdEdgeNode] = bm.einsum('ij,ik->ijk',grad1d,grad1d)
        if self.Normal2d is not None:
            normal = self.Normal2d(node)
            snormal = normal[isBdFaceNode]
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
        indices = bm.stack([rows.flat, cols.flat],axis=0)
        Pi = COOTensor(indices,Pi.flatten(),spshape=(3*NN, 3*NN))
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
    def build_preconditioner(
        self,
        update_interval: int = 3,
        rtol: float = 1e-2,
        maxiter: int = 100,
    ):
        return ProjectedCGPreconditioner(
            self,
            update_interval=update_interval,
            rtol=rtol,
            maxiter=maxiter,
        )
    def flipopt(self):
        node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        NN = self.mesh.number_of_nodes()
        self._NFlip +=1
        newcell3,oldcelltag2 = self._flip23()
        newcell2,oldcelltag3 = self._flip32()
        Nf = oldcelltag2.shape[0]
        Ne = oldcelltag3.shape[0]
        if Nf==0 and Ne==0:
            return False
        flip_cells = []
        flip_cells += [tuple(row) for row in oldcelltag2]
        flip_cells += [tuple(row) for row in oldcelltag3]
        
        used = set()
        keep = []
        
        for cells in flip_cells:
            if any(c in used for c in cells):
                keep.append(False)
            else:
                keep.append(True)
                used.update(cells)
        if Nf > 0:
            keepNf = bm.array(keep[:Nf])
            print('NF:',bm.sum(keepNf))
            oldcelltag2 = oldcelltag2[keepNf]
            newcell3 = newcell3[bm.repeat(keepNf,3)]
            cell[oldcelltag2[:,0]] = newcell3[::3]
            cell[oldcelltag2[:,1]] = newcell3[1::3]
            cell = bm.concatenate([cell, newcell3[2::3]],axis=0)
        if Ne>0: 
            keepNe = bm.array(keep[Nf:Nf+Ne])
            print('Ne:',bm.sum(keepNe))
            oldcelltag3 = oldcelltag3[keepNe]
            newcell2 = newcell2[bm.repeat(keepNe,2)]
            cell[oldcelltag3[:,0]] = newcell2[::2]
            cell[oldcelltag3[:,1]] = newcell2[1::2]
        if Ne>0:
            mask = bm.ones(cell.shape[0],dtype=bm.bool)
            mask[oldcelltag3[:,2]] = False
            cell = cell[mask]
        self.mesh.cell = cell
        self.mesh.construct()
        volume = self.mesh.entity_measure('cell')
        if bm.sum(volume<0)>0:
            print('badcell:',cell[volume<0])
            print('newcell3:',newcell3)
            print('newcell2:',newcell2)
        return True

    def _flip23(self):
        node = self.mesh.entity('node')
        face = self.mesh.entity('face')
        cell = self.mesh.entity('cell')
        NC = self.mesh.number_of_cells()
        face2cell = self.mesh.face_to_cell()
        faceable = bm.ones(face.shape[0],dtype=bool)
        faceable[face2cell[:,0]==face2cell[:,1]] = False
        c0 = face2cell[faceable,0]
        c1 = face2cell[faceable,1]
        lf0 = face2cell[faceable,2]
        lf1 = face2cell[faceable,3]
         
        f0 = face[faceable, :]
        n0 = cell[c0, lf0]
        n1 = cell[c1, lf1]
        oldcell2 = bm.zeros((faceable.sum()*2, 4), dtype=cell.dtype)
        newcell3 = bm.zeros((faceable.sum()*3, 4), dtype=cell.dtype)

        oldcell2[::2,:] = cell[c0]
        oldcell2[1::2,:] = cell[c1]
        newcell3[::3,0] = f0[:,0]
        newcell3[::3,1] = f0[:,1]
        newcell3[::3,2] = n0
        newcell3[::3,3] = n1
        newcell3[1::3,0] = f0[:,1]
        newcell3[1::3,1] = f0[:,2]
        newcell3[1::3,2] = n0
        newcell3[1::3,3] = n1
        newcell3[2::3,0] = f0[:,2]
        newcell3[2::3,1] = f0[:,0]
        newcell3[2::3,2] = n0
        newcell3[2::3,3] = n1
        v1 = node[newcell3[:,0]]-node[newcell3[:,1]]
        v2 = node[newcell3[:,0]]-node[newcell3[:,2]]
        v3 = node[newcell3[:,0]]-node[newcell3[:,3]]
        cm = bm.sum(-v3*bm.cross(v1,v2),axis=1)/6.0
        badcell = cm<=1e-14
        badface = badcell[::3]|badcell[1::3]|badcell[2::3]
        goodface = badcell[::3]&badcell[1::3]&badcell[2::3]
        temp_col2 = newcell3[goodface.repeat(3),2].copy()
        temp_col3 = newcell3[goodface.repeat(3),3].copy()
        newcell3[goodface.repeat(3), 2] = temp_col3
        newcell3[goodface.repeat(3), 3] = temp_col2
        badface[goodface] = False
        c0 = c0[~badface]
        c1 = c1[~badface]
        oldcell2 = oldcell2[~badface.repeat(2)]
        newcell3 = newcell3[~badface.repeat(3)]
        q1 = self._quality(oldcell2)
        q2 = self._quality(newcell3)
        
        if self._NFlip>4 or NC<100000:
            q1mean = (q1[::2]+q1[1::2])/2
            q2mean = (q2[::3]+q2[1::3]+q2[2::3])/3
            faceable = (q2mean<q1mean)#&(~badchose)
        else:
            q1max = bm.maximum(q1[::2], q1[1::2])
            q2max = bm.maximum(bm.maximum(q2[::3], q2[1::3]), q2[2::3])
            faceable = (q2max<q1max)#&(~badchose)

        c0 = c0[faceable]
        c1 = c1[faceable]
        faceablenew = bm.repeat(faceable,3)
        newcell3 = newcell3[faceablenew]
        return newcell3,bm.array([c0,c1]).T

    def _flip32(self):
        node = self.mesh.entity('node')
        edge = self.mesh.entity('edge')
        cell = self.mesh.entity('cell')
        edge2cell = self.mesh.edge_to_cell()
        boundary_edge_flag = self.mesh.boundary_edge_flag()
        numedge2cell = edge2cell.indptr[1:]-edge2cell.indptr[:-1] 
        edgeable = (numedge2cell==3)&(~boundary_edge_flag)
        edgeable = edgeable.flatten()
        edgelist = bm.where(edgeable)[0]
        starts = edge2cell.indptr[edgelist]
        edgecell = bm.stack([
                edge2cell.indices[starts],
                edge2cell.indices[starts+1],
                edge2cell.indices[starts+2]
                ],axis=1)
        Ne = edgelist.shape[0]
        if Ne == 0:
            return bm.array([]),bm.array([])
        verts = cell[edgecell].reshape(Ne,-1)
        ab = edge[edgelist]
        a = ab[:,0]
        b = ab[:,1]
        mask = (verts != a[:,None])&(verts != b[:,None])
        others = verts[mask].reshape(Ne,-1)
        others.sort(axis=1)
        face_nodes = others[:,::2]
        newcell2 = bm.zeros((Ne*2,4), dtype=cell.dtype)
        newcell2[::2,:3] = face_nodes
        newcell2[1::2,:3] = face_nodes
        newcell2[::2,3] = a
        newcell2[1::2,3] = b
        v1 = node[newcell2[:,0]]-node[newcell2[:,1]]
        v2 = node[newcell2[:,0]]-node[newcell2[:,2]]
        v3 = node[newcell2[:,0]]-node[newcell2[:,3]]
        cm = bm.sum(-v3*bm.cross(v1,v2),axis=1)/6.0
        neg = cm<0
        badcell = bm.abs(cm) <=1e-14
        badedge = badcell[::2]|badcell[1::2]
        temp_col2 = newcell2[neg, 2].copy()
        temp_col3 = newcell2[neg, 3].copy()
        newcell2[neg, 2] = temp_col3
        newcell2[neg, 3] = temp_col2
        oldcell3 = verts.reshape(-1,4)
        q1 = self._quality(oldcell3) 
        q2 = self._quality(newcell2)
        q1mean = (q1[::3]+q1[1::3]+q1[2::3])/3
        q2mean = (q2[::2]+q2[1::2])/2
        edgeable = q2mean<q1mean
        edgeable[badedge] = False
        edgeablenew = bm.repeat(edgeable,2)
        newcell2 = newcell2[edgeablenew]
        oldcelltag = edgecell[edgeable]
        return newcell2,oldcelltag

    def _flip44(self):
        node = self.mesh.entity('node')
        edge = self.mesh.entity('edge')
        cell = self.mesh.entity('cell')
        edge2cell = self.mesh.edge_to_cell()
        boundary_edge_flag = self.mesh.boundary_edge_flag()

        numedge2cell = bm.sum(edge2cell,axis=1).flatten()
        edgeable = (numedge2cell==4)&(~boundary_edge_flag)
        edgeable = edgeable.flatten()
        edgelist = bm.where(edgeable)[1]
        Ne = edgelist.shape[0]
        if Ne==0:
            return bm.array([]),bm.array([])
        start = edge2cell.indptr[edgelist]
        edgecell = bm.stack([
                edge2cell.indices[start],
                edge2cell.indices[start+1],
                edge2cell.indices[start+2],
                edge2cell.indices[start+3]
                ],axis=1)
        def orient_tets(tets):
            v1 = node[tets[:, 0]] - node[tets[:, 1]]
            v2 = node[tets[:, 0]] - node[tets[:, 2]]
            v3 = node[tets[:, 0]] - node[tets[:, 3]]
            cm = bm.sum(-v3 * bm.cross(v1, v2), axis=1) / 6.0
            neg = cm < 0
            temp_col2 = tets[neg, 2].copy()
            temp_col3 = tets[neg, 3].copy()
            tets[neg, 2] = temp_col3
            tets[neg, 3] = temp_col2
            v1 = node[tets[:, 0]] - node[tets[:, 1]]
            v2 = node[tets[:, 0]] - node[tets[:, 2]]
            v3 = node[tets[:, 0]] - node[tets[:, 3]]
            cm = bm.sum(-v3 * bm.cross(v1, v2), axis=1) / 6.0
            return tets
        verts = cell[edgecell].reshape(Ne,-1)
        uv = edge[edgelist]
        ac = bm.zeros_like(uv,dtype=uv.dtype)
        bd = bm.zeros_like(uv,dtype=uv.dtype)
        u = uv[:,0]
        v = uv[:,1]
        mask = (verts != u[:,None])&(verts != v[:,None])
        others = verts[mask].reshape(Ne,-1)
        flag1 = (others == others[:,0,None])
        cols = bm.where(flag1)[1][1::2]
        rows = bm.arange(Ne)
        flag2 = cols%2==0
        cols[flag2] = cols[flag2]+1
        cols[~flag2] = cols[~flag2]-1
        ac[:,0] = others[:,0]
        ac[:,1] = others[rows,cols]
        mask = (others != ac[:,0,None])&(others != ac[:,1,None])
        others = others[mask].reshape(Ne,-1)
        others.sort(axis=1)
        bd[:,0] = others[:,0]
        bd[:,1] = others[:,2]
        indx = bm.arange(Ne*4)
        indx0 = indx.reshape(-1,4)[:,:2].reshape(-1)
        indx1 = indx.reshape(-1,4)[:,2:].reshape(-1)
        newcell4_1 = bm.zeros((Ne*4,4), dtype=cell.dtype)
        newcell4_2 = bm.zeros((Ne*4,4), dtype=cell.dtype)
        newcell4_1[::2,0] = u.repeat(2)
        newcell4_1[1::2,0] = v.repeat(2)
        newcell4_1[:,1] = ac[:,0].repeat(4)
        newcell4_1[indx0,2] = bd[:,0].repeat(2)
        newcell4_1[indx1,2] = ac[:,1].repeat(2)
        newcell4_1[indx0,3] = ac[:,1].repeat(2)
        newcell4_1[indx1,3] = bd[:,1].repeat(2)
        newcell4_2[::2,0] = u.repeat(2)
        newcell4_2[1::2,0] = v.repeat(2)
        newcell4_2[indx0,1] = ac[:,0].repeat(2)
        newcell4_2[indx1,1] = bd[:,0].repeat(2)
        newcell4_2[indx0,2] = bd[:,0].repeat(2)
        newcell4_2[indx1,2] = ac[:,1].repeat(2)
        newcell4_2[:,3] = bd[:,1].repeat(4)
        newcell4_1 = orient_tets(newcell4_1)
        newcell4_2 = orient_tets(newcell4_2)
        oldcell4 = verts.reshape(-1,4)
        q1 = self._quality(oldcell4)
        q2 = self._quality(newcell4_1)
        q3 = self._quality(newcell4_2)
        q1mean = (q1[::4]+q1[1::4]+q1[2::4]+q1[3::4])/4
        q2mean = (q2[::4]+q2[1::4]+q2[2::4]+q2[3::4])/4
        q3mean = (q3[::4]+q3[1::4]+q3[2::4]+q3[3::4])/4
        bestmean = q1mean
        newcell4 = oldcell4
        edgeable1 = (q2mean<bestmean)
        newcell4[edgeable1.repeat(4)] = newcell4_1[edgeable1.repeat(4)]
        bestmean[edgeable1] = q2mean[edgeable1]
        edgeable2 = (q3mean<bestmean)
        newcell4[edgeable2.repeat(4)] = newcell4_2[edgeable2.repeat(4)]
        edgeable = edgeable1|edgeable2
        newcell4 = newcell4[edgeable.repeat(4)]
        oldcelltag = edgecell[edgeable]
        return newcell4,oldcelltag

    def _quality(self,cell):
        node = self.mesh.entity('node')

        v10 = node[cell[:, 0]] - node[cell[:, 1]]
        v20 = node[cell[:, 0]] - node[cell[:, 2]]
        v30 = node[cell[:, 0]] - node[cell[:, 3]]
        v102 = bm.sum(v10**2, axis=-1)
        v202 = bm.sum(v20**2, axis=-1)
        v302 = bm.sum(v30**2, axis=-1)

        d = v302[:, None]*(bm.cross(v10, v20)) + v102[:, None]*(bm.cross(v20, v30)
                ) + v202[:, None]*(bm.cross(v30, v10))
        dl = bm.sqrt(bm.sum(d**2, axis=-1))
        
        face0 = bm.array([cell[:,0],cell[:,1],cell[:,2]]).T
        face1 = bm.array([cell[:,0],cell[:,1],cell[:,3]]).T
        face2 = bm.array([cell[:,0],cell[:,2],cell[:,3]]).T
        face3 = bm.array([cell[:,1],cell[:,2],cell[:,3]]).T

        v01 = node[face0[:, 1], :] - node[face0[:, 0], :]
        v02 = node[face0[:, 2], :] - node[face0[:, 0], :]
        fm0 = bm.sqrt(bm.square(bm.cross(v01,v02)).sum(axis=1))/2.0

        v01 = node[face1[:, 1], :] - node[face1[:, 0], :]
        v02 = node[face1[:, 2], :] - node[face1[:, 0], :]
        fm1 = bm.sqrt(bm.square(bm.cross(v01,v02)).sum(axis=1))/2.0
        
        v01 = node[face2[:, 1], :] - node[face2[:, 0], :]
        v02 = node[face2[:, 2], :] - node[face2[:, 0], :]
        fm2 = bm.sqrt(bm.square(bm.cross(v01,v02)).sum(axis=1))/2.0

        v01 = node[face3[:, 1], :] - node[face3[:, 0], :]
        v02 = node[face3[:, 2], :] - node[face3[:, 0], :]
        fm3 = bm.sqrt(bm.square(bm.cross(v01,v02)).sum(axis=1))/2.0

        cm = bm.sum(-v30*bm.cross(v10,v20),axis=1)/6.0
        
        s_sum = fm0+fm1+fm2+fm3
        quality = s_sum*dl/108/cm/cm
        penalty = self.penalty(cm)
        quality = quality+penalty
        return quality

    @staticmethod
    def get_quality(mesh):
        GD = mesh.geo_dimension()
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        
        v10 = node[cell[:, 0]] - node[cell[:, 1]]
        v20 = node[cell[:, 0]] - node[cell[:, 2]]
        v30 = node[cell[:, 0]] - node[cell[:, 3]]
        v21 = node[cell[:, 1]] - node[cell[:, 2]]
        v31 = node[cell[:, 1]] - node[cell[:, 3]]
        v32 = node[cell[:, 2]] - node[cell[:, 3]]

        l10 = bm.sum(v10**2, axis=-1)
        l20 = bm.sum(v20**2, axis=-1)
        l30 = bm.sum(v30**2, axis=-1)
        l21 = bm.sum(v21**2, axis=-1)
        l31 = bm.sum(v31**2, axis=-1)
        l32 = bm.sum(v32**2, axis=-1)

        d0 = bm.zeros((NC, 3), dtype=mesh.ftype)
        c12 =  bm.cross(v10, v20)
        d0 += l30[:, None]*c12
        c23 = bm.cross(v20, v30)
        d0 += l10[:, None]*c23
        c31 = bm.cross(v30, v10)
        d0 += l20[:, None]*c31

        ld0 = bm.sum(d0**2, axis=-1)

        face = mesh.entity('face')
        fv01 = node[face[:, 1], :] - node[face[:, 0], :]
        fv02 = node[face[:, 2], :] - node[face[:, 0], :]
        fm = bm.sqrt(bm.square(bm.cross(fv01,fv02)).sum(axis=1))/2.0

        cm = bm.sum(-v30*bm.cross(v10,v20),axis=1)/6.0
        c2f = mesh.cell_to_face()
        
        s = fm[c2f]
        s_sum = bm.sum(s, axis=-1)
        quality = s_sum*bm.sqrt(ld0)/(108*cm*cm)
        return quality

