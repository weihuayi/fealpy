from typing import TypedDict, Callable, Tuple, Union, Optional

from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike 
from fealpy.mesh import TetrahedronMesh
from fealpy.sparse import COOTensor,CSRTensor

from opt import Problem
from .rropreconditioner import ProjectedCGPreconditioner


class TetAniMeshProblem(Problem):
    def __init__(self,options: dict):
        self.mesh = options["mesh"]
        node = self.mesh.entity('node')
        if options["isFixNode"] is None:
            self.isFixNode = bm.zeros(node.shape[0], dtype=bm.bool)
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
        self.Metric = options["Metric"]
        self.m = self.metric(node)
        self.mc,self.mcinv = self.cell_metric(self.m)
        mc = self.mc
        self.mcinvs = self.inv_sqrt_metric(self.mc)
        self.detmc = mc[:,0,0]*mc[:,1,1]*mc[:,2,2]+2*mc[:,0,1]*mc[:,1,2]*mc[:,0,2]-\
                mc[:,0,0]*mc[:,1,2]*mc[:,1,2]-mc[:,0,1]*mc[:,0,1]*mc[:,2,2]-\
                mc[:,1,1]*mc[:,0,2]*mc[:,0,2]
        self.sdetmc = bm.sqrt(self.detmc)
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
            Normal2d: Optional[Callable[[TensorLike], TensorLike]] = None,
            Metric: Optional[Callable[[TensorLike], TensorLike]] = None 
            ) -> dict:
        options = {
            'mesh': mesh,
            'FixAllBoundary': FixAllBoundary,
            'isFixNode': isFixNode,
            'isBdEdgeNode': isBdEdgeNode,
            'isBdFaceNode': isBdFaceNode,
            'Project': Project,
            'Tangent1d': Tangent1d,
            'Normal2d': Normal2d,
            'Metric': Metric
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

    def metric(self,node):
        M = self.Metric(node)
        return M

    def cell_metric(self,M):
        cell = self.mesh.entity('cell')
        MC = (M[cell[:,0]]+M[cell[:,1]]+M[cell[:,2]]+M[cell[:,3]])/4
        detMC = MC[:,0,0]*MC[:,1,1]*MC[:,2,2]+2*MC[:,0,1]*MC[:,1,2]*MC[:,0,2]-\
        MC[:,0,0]*MC[:,1,2]*MC[:,1,2]-MC[:,0,1]*MC[:,0,1]*MC[:,2,2]-\
        MC[:,1,1]*MC[:,0,2]*MC[:,0,2]
         
        MCinv = bm.zeros_like(MC)
       
        MCinv[:,0,0] = MC[:,1,1]*MC[:,2,2]-MC[:,1,2]*MC[:,1,2]
        MCinv[:,0,1] = MC[:,0,2]*MC[:,1,2]-MC[:,0,1]*MC[:,2,2]
        MCinv[:,0,2] = MC[:,0,1]*MC[:,1,2]-MC[:,0,2]*MC[:,1,1]
        MCinv[:,1,1] = MC[:,0,0]*MC[:,2,2]-MC[:,0,2]*MC[:,0,2]
        MCinv[:,1,2] = MC[:,0,1]*MC[:,0,2]-MC[:,0,0]*MC[:,1,2]
        MCinv[:,2,2] = MC[:,0,0]*MC[:,1,1]-MC[:,0,1]*MC[:,0,1]
        MCinv[:,1,0] = MCinv[:,0,1]
        MCinv[:,2,0] = MCinv[:,0,2]
        MCinv[:,2,1] = MCinv[:,1,2]
        MCinv /= detMC[:,None,None]
        return MC,MCinv

    def inv_sqrt_metric(self,M):
        w, V = bm.linalg.eigh(M)
        w_inv_sqrt = 1.0/bm.sqrt(w)
        Minv_sqrt = bm.einsum('...ik,...k,...jk->...ij', V, w_inv_sqrt, V)
        return Minv_sqrt

    def penalty(self,q):
        pen = bm.zeros_like(q)
        flag = q<0
        pen[flag] = (1-q[flag]/1e-8)**2
        return pen

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

        NC = self.mesh.number_of_cells()
        NN = self.mesh.number_of_nodes()
        
        m = self.metric(node)
        mc,mcinv = self.cell_metric(m)

        m00 = mc[:,0,0]
        m11 = mc[:,1,1]
        m22 = mc[:,2,2]
        m01 = mc[:,0,1]
        m02 = mc[:,0,2]
        m12 = mc[:,1,2]

        mcinvs = self.inv_sqrt_metric(mc)
        detmc = mc[:,0,0]*mc[:,1,1]*mc[:,2,2]+2*mc[:,0,1]*mc[:,1,2]*mc[:,0,2]-\
                mc[:,0,0]*mc[:,1,2]*mc[:,1,2]-mc[:,0,1]*mc[:,0,1]*mc[:,2,2]-\
                mc[:,1,1]*mc[:,0,2]*mc[:,0,2]
        sdetmc = bm.sqrt(detmc)

        n00 = mc[:,0,0]/detmc
        n11 = mc[:,1,1]/detmc
        n22 = mc[:,2,2]/detmc
        n01 = mc[:,0,1]/detmc
        n02 = mc[:,0,2]/detmc
        n12 = mc[:,1,2]/detmc

        v10 = node[cell[:, 0]] - node[cell[:, 1]]
        v20 = node[cell[:, 0]] - node[cell[:, 2]]
        v30 = node[cell[:, 0]] - node[cell[:, 3]]
        v21 = node[cell[:, 1]] - node[cell[:, 2]]
        v31 = node[cell[:, 1]] - node[cell[:, 3]]
        v32 = node[cell[:, 2]] - node[cell[:, 3]]
        
        l10 = bm.einsum('ni,nij,nj->n', v10, mc, v10)
        l20 = bm.einsum('ni,nij,nj->n', v20, mc, v20)
        l30 = bm.einsum('ni,nij,nj->n', v30, mc, v30)
        l21 = bm.einsum('ni,nij,nj->n', v21, mc, v21)
        l31 = bm.einsum('ni,nij,nj->n', v31, mc, v31)
        l32 = bm.einsum('ni,nij,nj->n', v32, mc, v32)
        l31_30 = bm.einsum('ni,nij,nj->n', v31, mc, v30)
        l32_30 = bm.einsum('ni,nij,nj->n', v32, mc, v30)
        l21_20 = bm.einsum('ni,nij,nj->n', v21, mc, v20)
        l21_10 = bm.einsum('ni,nij,nj->n', v21, mc, v10)
        l31_10 = bm.einsum('ni,nij,nj->n', v31, mc, v10)
        l32_31 = bm.einsum('ni,nij,nj->n', v32, mc, v31)
        l20_10 = bm.einsum('ni,nij,nj->n', v20, mc, v10)
        l32_20 = bm.einsum('ni,nij,nj->n', v32, mc, v20)
        l30_10 = bm.einsum('ni,nij,nj->n', v30, mc, v10)
        l30_20 = bm.einsum('ni,nij,nj->n', v30, mc, v20)
        l31_21 = bm.einsum('ni,nij,nj->n', v31, mc, v21)
        l32_21 = bm.einsum('ni,nij,nj->n', v32, mc, v21)

        d0 = bm.zeros((NC, 3), dtype=self.mesh.ftype)
        
        c12 = bm.einsum('ijk,ik->ij',mcinvs,bm.cross(v10,v20))
        d0 += l30[:, None]*c12
        c23 = bm.einsum('ijk,ik->ij',mcinvs,bm.cross(v20,v30))
        d0 += l10[:, None]*c23
        c31 = bm.einsum('ijk,ik->ij',mcinvs,bm.cross(v30,v10))
        d0 += l20[:, None]*c31
        
        d0x = bm.einsum('ijk,ik->ij',mcinvs,d0)
        ld0 = bm.sum(d0**2,axis=-1)

        c12 = bm.sum(c12*d0, axis=-1)
        c23 = bm.sum(c23*d0, axis=-1)
        c31 = bm.sum(c31*d0, axis=-1)
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

        S = bm.zeros((NC, 4, 4), dtype=self.mesh.ftype)
        face = self.mesh.entity('face')
        fv01 = node[face[:, 1], :] - node[face[:, 0], :]
        fv02 = node[face[:, 2], :] - node[face[:, 0], :]
        vf = bm.cross(fv01,fv02)
        
        cm = bm.sum(-v30*bm.cross(v10,v20),axis=1)/6.0
        c2f = self.mesh.cell_to_face()
        vf0 = bm.einsum('ijk,ik->ij',mcinvs,vf[c2f[:,0]])
        vf1 = bm.einsum('ijk,ik->ij',mcinvs,vf[c2f[:,1]])
        vf2 = bm.einsum('ijk,ik->ij',mcinvs,vf[c2f[:,2]])
        vf3 = bm.einsum('ijk,ik->ij',mcinvs,vf[c2f[:,3]])

        s0 = bm.sqrt(bm.square(vf0).sum(axis=1))/2.0
        s1 = bm.sqrt(bm.square(vf1).sum(axis=1))/2.0
        s2 = bm.sqrt(bm.square(vf2).sum(axis=1))/2.0
        s3 = bm.sqrt(bm.square(vf3).sum(axis=1))/2.0
        s_sum = s0+s1+s2+s3
        quality = s_sum*bm.sqrt(ld0)/(108*cm**2)
        penalty = self.penalty(cm)
        quality = quality+penalty

        p0 = (l31/s2 + l21/s3 + l32/s1)/4
        p1 = (l32/s0 + l20/s3 + l30/s2)/4
        p2 = (l30/s1 + l10/s3 + l31/s0)/4
        p3 = (l10/s2 + l20/s1 + l21/s0)/4

        q10 = -(l31_30/s2+l21_20/s3)/4
        q20 = -(l32_30/s1-l21_10/s3)/4
        q30 = -(-l32_20/s1-l31_10/s2)/4
        q21 = -(l32_31/s0+l20_10/s3)/4
        q31 = -(l30_10/s2-l32_21/s0)/4
        q32 = -(l31_21/s0+l30_20/s1)/4

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

        K0 = -d0x[:,0,None,None]*K
        K1 = d0x[:,1,None,None]*K
        K2 = -d0x[:,2,None,None]*K
        ld0 = bm.sum(d0**2,axis=-1)

        A  /= ld0[:,None,None]
        K0 /= ld0[:,None,None]
        K1 /= ld0[:,None,None]
        K2 /= ld0[:,None,None]
        
        S /= s_sum[:,None,None]

        C0 /= 3*cm[:,None,None]
        C1 /= 3*cm[:,None,None]
        C2 /= 3*cm[:,None,None]
        
        A0 = m00[:,None,None]*A + n00[:,None,None]*S
        A1 = m11[:,None,None]*A + n11[:,None,None]*S
        A2 = m22[:,None,None]*A + n22[:,None,None]*S
        B0 = m12[:,None,None]*A + K0 + n12[:,None,None]*S-C0
        B0T= m12[:,None,None]*A - K0 + n12[:,None,None]*S+C0
        B1 = m02[:,None,None]*A + K1 + n02[:,None,None]*S-C1
        B1T= m02[:,None,None]*A - K1 + n02[:,None,None]*S+C1
        B2 = m01[:,None,None]*A + K2 + n01[:,None,None]*S-C2
        B2T= m01[:,None,None]*A - K2 + n01[:,None,None]*S+C2
        
        A0 *= quality[:,None,None]/NC
        A1 *= quality[:,None,None]/NC
        A2 *= quality[:,None,None]/NC
        B0 *= quality[:,None,None]/NC
        B1 *= quality[:,None,None]/NC
        B2 *= quality[:,None,None]/NC
        B0T *= quality[:,None,None]/NC
        B1T *= quality[:,None,None]/NC
        B2T *= quality[:,None,None]/NC
        
        I = bm.broadcast_to(cell[:, :, None], (NC, 4, 4))
        J = bm.broadcast_to(cell[:, None, :], (NC, 4, 4))
        indices = bm.stack([I.flatten(),J.flatten()],axis=0)
        A0 = COOTensor(indices,A0.flatten(),spshape=(NN,NN))
        A1 = COOTensor(indices,A1.flatten(),spshape=(NN,NN))
        A2 = COOTensor(indices,A2.flatten(),spshape=(NN,NN))
        B0 = COOTensor(indices,B0.flatten(),spshape=(NN,NN))
        B1 = COOTensor(indices,B1.flatten(),spshape=(NN,NN))
        B2 = COOTensor(indices,B2.flatten(),spshape=(NN,NN))
        B0T = COOTensor(indices,B0T.flatten(),spshape=(NN,NN))
        B1T = COOTensor(indices,B1T.flatten(),spshape=(NN,NN))
        B2T = COOTensor(indices,B2T.flatten(),spshape=(NN,NN))

        gradp = bm.full_like(node,0.0)
        gradp[:,0] = (A0@node[:,0]+B2@node[:,1]+B1@node[:,2])
        gradp[:,1] = (B2T@node[:,0]+A1@node[:,1]+B0@node[:,2])
        gradp[:,2] = (B1T@node[:,0]+B0T@node[:,1]+A2@node[:,2])
        gradp = gradp[~self.isFixNode]
        gradp = self.tangent_project(gradp,x)
        gradp = gradp.T.flatten()
        return bm.mean(quality), gradp

    @classmethod
    def get_quality(cls,mesh,metric):
        options = cls.get_options(mesh=mesh,Metric=metric)
        problem = cls(options)
        GD = mesh.geo_dimension()
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        
        m = problem.metric(node)
        mc,mcinv = problem.cell_metric(m)
        mcinvs = problem.inv_sqrt_metric(mc)

        detmc = mc[:,0,0]*mc[:,1,1]*mc[:,2,2]+2*mc[:,0,1]*mc[:,1,2]*mc[:,0,2]-\
                mc[:,0,0]*mc[:,1,2]*mc[:,1,2]-mc[:,1,1]*mc[:,0,2]*mc[:,0,2]-\
                mc[:,2,2]*mc[:,0,1]*mc[:,0,1]
        
        v10 = node[cell[:, 0]] - node[cell[:, 1]]
        v20 = node[cell[:, 0]] - node[cell[:, 2]]
        v30 = node[cell[:, 0]] - node[cell[:, 3]]

        v21 = node[cell[:, 1]] - node[cell[:, 2]]
        v31 = node[cell[:, 1]] - node[cell[:, 3]]
        v32 = node[cell[:, 2]] - node[cell[:, 3]]
        
        l10 = mc[:,0,0]*v10[:,0]*v10[:,0]+mc[:,1,1]*v10[:,1]*v10[:,1]+\
              mc[:,2,2]*v10[:,2]*v10[:,2]+2*(mc[:,0,1]*v10[:,0]*v10[:,1]+\
              mc[:,0,2]*v10[:,0]*v10[:,2]+mc[:,1,2]*v10[:,1]*v10[:,2])
        l20 = mc[:,0,0]*v20[:,0]*v20[:,0]+mc[:,1,1]*v20[:,1]*v20[:,1]+\
              mc[:,2,2]*v20[:,2]*v20[:,2]+2*(mc[:,0,1]*v20[:,0]*v20[:,1]+\
              mc[:,0,2]*v20[:,0]*v20[:,2]+mc[:,1,2]*v20[:,1]*v20[:,2])
        l30 = mc[:,0,0]*v30[:,0]*v30[:,0]+mc[:,1,1]*v30[:,1]*v30[:,1]+\
              mc[:,2,2]*v30[:,2]*v30[:,2]+2*(mc[:,0,1]*v30[:,0]*v30[:,1]+\
              mc[:,0,2]*v30[:,0]*v30[:,2]+mc[:,1,2]*v30[:,1]*v30[:,2])
        l21 = mc[:,0,0]*v21[:,0]*v21[:,0]+mc[:,1,1]*v21[:,1]*v21[:,1]+\
              mc[:,2,2]*v21[:,2]*v21[:,2]+2*(mc[:,0,1]*v21[:,0]*v21[:,1]+\
              mc[:,0,2]*v21[:,0]*v21[:,2]+mc[:,1,2]*v21[:,1]*v21[:,2])
        l31 = mc[:,0,0]*v31[:,0]*v31[:,0]+mc[:,1,1]*v31[:,1]*v31[:,1]+\
              mc[:,2,2]*v31[:,2]*v31[:,2]+2*(mc[:,0,1]*v31[:,0]*v31[:,1]+\
              mc[:,0,2]*v31[:,0]*v31[:,2]+mc[:,1,2]*v31[:,1]*v31[:,2])
        l32 = mc[:,0,0]*v32[:,0]*v32[:,0]+mc[:,1,1]*v32[:,1]*v32[:,1]+\
              mc[:,2,2]*v32[:,2]*v32[:,2]+2*(mc[:,0,1]*v32[:,0]*v32[:,1]+
              mc[:,0,2]*v32[:,0]*v32[:,2]+mc[:,1,2]*v32[:,1]*v32[:,2])

        d0 = bm.zeros((NC, 3), dtype=mesh.ftype)
        
        c12 = bm.einsum('ijk,ik->ij',mcinvs,bm.cross(v10,v20))
        d0 += l30[:, None]*c12
        c23 = bm.einsum('ijk,ik->ij',mcinvs,bm.cross(v20,v30))
        d0 += l10[:, None]*c23
        c31 = bm.einsum('ijk,ik->ij',mcinvs,bm.cross(v30,v10))
        d0 += l20[:, None]*c31
        ld0 = bm.sum(d0**2,axis=-1)

        face = mesh.entity('face')
        fv01 = node[face[:, 1], :] - node[face[:, 0], :]
        fv02 = node[face[:, 2], :] - node[face[:, 0], :]
        vf = bm.cross(fv01,fv02)
        
        cm = bm.sum(-v30*bm.cross(v10,v20),axis=1)/6.0
        c2f = mesh.cell_to_face()
        vf0 = bm.einsum('ijk,ik->ij',mcinvs,vf[c2f[:,0]])
        vf1 = bm.einsum('ijk,ik->ij',mcinvs,vf[c2f[:,1]])
        vf2 = bm.einsum('ijk,ik->ij',mcinvs,vf[c2f[:,2]])
        vf3 = bm.einsum('ijk,ik->ij',mcinvs,vf[c2f[:,3]])
        
        s0 = bm.sqrt(bm.square(vf0).sum(axis=1))/2
        s1 = bm.sqrt(bm.square(vf1).sum(axis=1))/2
        s2 = bm.sqrt(bm.square(vf2).sum(axis=1))/2
        s3 = bm.sqrt(bm.square(vf3).sum(axis=1))/2
        s_sum = s0+s1+s2+s3

        quality = s_sum*bm.sqrt(ld0)/(108*cm**2)
        penalty = problem.penalty(cm)
        quality = quality+penalty
        return quality

    def flipopt(self):
        node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        NN = self.mesh.number_of_nodes()
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
            oldcelltag2 = oldcelltag2[keepNf]
            newcell3 = newcell3[bm.repeat(keepNf,3)]
            cell[oldcelltag2[:,0]] = newcell3[::3]
            cell[oldcelltag2[:,1]] = newcell3[1::3]
            cell = bm.concatenate([cell, newcell3[2::3]],axis=0)
        if Ne>0: 
            keepNe = bm.array(keep[Nf:])
            oldcelltag3 = oldcelltag3[keepNe]
            newcell2 = newcell2[bm.repeat(keepNe,2)]
            cell[oldcelltag3[:,0]] = newcell2[::2]
            cell[oldcelltag3[:,1]] = newcell2[1::2]
            mask = bm.ones(cell.shape[0],dtype=bm.bool)
            mask[oldcelltag3[:,2]] = False
            cell = cell[mask]
        self.mesh.cell = cell
        self.mesh.construct()
        return True

    def _flip23(self):
        face = self.mesh.entity('face')
        cell = self.mesh.entity('cell')
        face2cell = self.mesh.face_to_cell()
        faceable = bm.ones(face.shape[0],dtype=bm.bool)
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
        q1 = self._quality(oldcell2)
        q2 = self._quality(newcell3)
        q1mean = (q1[::2]+q1[1::2])/2
        q2mean = (q2[::3]+q2[1::3]+q2[2::3])/3
        faceable = q2mean<q1mean
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
        edgeablenew = bm.repeat(edgeable,2)
        newcell2 = newcell2[edgeablenew]
        oldcelltag = edgecell[edgeable]
        return newcell2,oldcelltag

    def _quality(self,cell):
        node = self.mesh.entity('node')
        M = self.metric(node)
        MC = (M[cell[:,0]]+M[cell[:,1]]+M[cell[:,2]]+M[cell[:,3]])/4
        detMC = MC[:,0,0]*MC[:,1,1]*MC[:,2,2]+2*MC[:,0,1]*MC[:,1,2]*MC[:,0,2]-\
        MC[:,0,0]*MC[:,1,2]*MC[:,1,2]-MC[:,0,1]*MC[:,0,1]*MC[:,2,2]-\
        MC[:,1,1]*MC[:,0,2]*MC[:,0,2]
        MCinv = bm.zeros_like(MC)

        MCinv[:,0,0] = MC[:,1,1]*MC[:,2,2]-MC[:,1,2]*MC[:,1,2]
        MCinv[:,0,1] = MC[:,0,2]*MC[:,1,2]-MC[:,0,1]*MC[:,2,2]
        MCinv[:,0,2] = MC[:,0,1]*MC[:,1,2]-MC[:,0,2]*MC[:,1,1]
        MCinv[:,1,1] = MC[:,0,0]*MC[:,2,2]-MC[:,0,2]*MC[:,0,2]
        MCinv[:,1,2] = MC[:,0,1]*MC[:,0,2]-MC[:,0,0]*MC[:,1,2]
        MCinv[:,2,2] = MC[:,0,0]*MC[:,1,1]-MC[:,0,1]*MC[:,0,1]
        MCinv[:,1,0] = MCinv[:,0,1]
        MCinv[:,2,0] = MCinv[:,0,2]
        MCinv[:,2,1] = MCinv[:,1,2]
        MCinv /= detMC[:,None,None]

        mcinvs = self.inv_sqrt_metric(MC)

        v10 = node[cell[:, 0]] - node[cell[:, 1]]
        v20 = node[cell[:, 0]] - node[cell[:, 2]]
        v30 = node[cell[:, 0]] - node[cell[:, 3]]
        v21 = node[cell[:, 1]] - node[cell[:, 2]]
        v31 = node[cell[:, 1]] - node[cell[:, 3]]
        v32 = node[cell[:, 2]] - node[cell[:, 3]]

        l10 = bm.einsum('ni,nij,nj->n', v10, MC, v10)
        l20 = bm.einsum('ni,nij,nj->n', v20, MC, v20)
        l30 = bm.einsum('ni,nij,nj->n', v30, MC, v30)
        l21 = bm.einsum('ni,nij,nj->n', v21, MC, v21)
        l31 = bm.einsum('ni,nij,nj->n', v31, MC, v31)
        l32 = bm.einsum('ni,nij,nj->n', v32, MC, v32)
        l31_30 = bm.einsum('ni,nij,nj->n', v31, MC, v30)
        l32_30 = bm.einsum('ni,nij,nj->n', v32, MC, v30)
        l21_20 = bm.einsum('ni,nij,nj->n', v21, MC, v20)
        l21_10 = bm.einsum('ni,nij,nj->n', v21, MC, v10)
        l31_10 = bm.einsum('ni,nij,nj->n', v31, MC, v10)
        l32_31 = bm.einsum('ni,nij,nj->n', v32, MC, v31)
        l20_10 = bm.einsum('ni,nij,nj->n', v20, MC, v10)
        l32_20 = bm.einsum('ni,nij,nj->n', v32, MC, v20)
        l30_10 = bm.einsum('ni,nij,nj->n', v30, MC, v10)
        l30_20 = bm.einsum('ni,nij,nj->n', v30, MC, v20)
        l31_21 = bm.einsum('ni,nij,nj->n', v31, MC, v21)
        l32_21 = bm.einsum('ni,nij,nj->n', v32, MC, v21)

        d0 = bm.zeros((cell.shape[0], 3), dtype=self.mesh.ftype)
        c12 = bm.einsum('ijk,ik->ij',mcinvs,bm.cross(v10,v20))
        d0 += l30[:, None]*c12
        c23 = bm.einsum('ijk,ik->ij',mcinvs,bm.cross(v20,v30))
        d0 += l10[:, None]*c23
        c31 = bm.einsum('ijk,ik->ij',mcinvs,bm.cross(v30,v10))
        d0 += l20[:, None]*c31
        ld0 = bm.sum(d0**2,axis=-1)

        face0 = bm.array([cell[:,0],cell[:,1],cell[:,2]]).T
        face1 = bm.array([cell[:,0],cell[:,1],cell[:,3]]).T
        face2 = bm.array([cell[:,0],cell[:,2],cell[:,3]]).T
        face3 = bm.array([cell[:,1],cell[:,2],cell[:,3]]).T

        fv01 = node[face0[:, 1], :] - node[face0[:, 0], :]
        fv02 = node[face0[:, 2], :] - node[face0[:, 0], :]
        vf0 = bm.einsum('ijk,ik->ij',mcinvs,bm.cross(fv01,fv02))

        fv01 = node[face1[:, 1], :] - node[face1[:, 0], :]
        fv02 = node[face1[:, 2], :] - node[face1[:, 0], :]
        vf1 = bm.einsum('ijk,ik->ij',mcinvs,bm.cross(fv01,fv02))

        fv01 = node[face2[:, 1], :] - node[face2[:, 0], :]
        fv02 = node[face2[:, 2], :] - node[face2[:, 0], :]
        vf2 = bm.einsum('ijk,ik->ij',mcinvs,bm.cross(fv01,fv02))

        fv01 = node[face3[:, 1], :] - node[face3[:, 0], :]
        fv02 = node[face3[:, 2], :] - node[face3[:, 0], :]
        vf3 = bm.einsum('ijk,ik->ij',mcinvs,bm.cross(fv01,fv02))

        s0 = bm.sqrt(bm.square(vf0).sum(axis=1))/2.0
        s1 = bm.sqrt(bm.square(vf1).sum(axis=1))/2.0
        s2 = bm.sqrt(bm.square(vf2).sum(axis=1))/2.0
        s3 = bm.sqrt(bm.square(vf3).sum(axis=1))/2.0

        cm = bm.sum(-v30*bm.cross(v10,v20),axis=1)/6.0
        
        s_sum = s0+s1+s2+s3
        quality = s_sum*bm.sqrt(ld0)/108/cm/cm
        penalty = self.penalty(cm)
        quality = quality+penalty
        return quality
