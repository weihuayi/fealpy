from typing import TypedDict, Callable, Tuple, Union, Optional

from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike 
from fealpy.mesh import TriangleMesh
from fealpy.sparse import COOTensor,CSRTensor

from opt import Problem

class TriAniMeshProblem(Problem):
    def __init__(self, options: dict):
        self.mesh = options["mesh"]
        node = self.mesh.entity('node')
        if options["isFixNode"] is None:
            self.isFixNode = bm.zeros(node.shape[0], dtype=bool)
        else:
            self.isFixNode = options["isFixNode"]
        self.FixNode = node[self.isFixNode]
        self.isFreeNode = ~self.isFixNode
        self.isBdEdgeNode = options["isBdEdgeNode"]
        self.isBdNode = self.mesh.boundary_node_flag()
        if self.isFixNode is not None:
            self.isBdNode[self.isFixNode] = False
            x0 = bm.array(node[self.isFreeNode].T.flat) 
        else:
            x0 = bm.array(node.T.flat)
        self.Project = options["Project"] 
        self.Tangent1d = options["Tangent1d"]
        self.Metric = options["Metric"]
        self.m = self.metric(node)
        self.mc = self.cell_metric(self.m)
        super().__init__(x0,self.quality)
        self.project_to_boundary = (
                self.project_to_bd if self.Project is not None else None
                )

    @classmethod
    def get_options(
        cls,*,
        mesh:TriangleMesh,
        Metric: Optional[Callable[[TensorLike],TensorLike]] = None,
        FixAllBoundary: bool = False,
        isFixNode: Optional[TensorLike] = None,
        isBdEdgeNode: Optional[TensorLike] = None,
        Project: Optional[Callable[[TensorLike],TensorLike]] = None,
        Tangent1d: Optional[Callable[[TensorLike, TensorLike], TensorLike]] = None,
        ) -> dict:
        options = {
            'mesh': mesh,
            'Metric':Metric,
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

    def metric(self,node):
        M = self.Metric(node)
        return M

    def cell_metric(self,M):
        cell = self.mesh.entity('cell')
        idxi = cell[:, 0]
        idxj = cell[:, 1] 
        idxk = cell[:, 2]
        Mi = M[idxi]
        Mj = M[idxj]
        Mk = M[idxk]
        MC = (Mi+Mj+Mk)/3
        return MC

    def quality(self, x):
        GD =self.mesh.geo_dimension()
        node0 = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        node = bm.full_like(node0,0.0)
        node[self.isFixNode] = self.FixNode
        NN = self.mesh.number_of_nodes()
        NC = self.mesh.number_of_cells()

        n = len(x)//2
        node[self.isFreeNode,0] = x[:n]
        node[self.isFreeNode,1] = x[n:]

        m = self.metric(node)
        mc = self.cell_metric(m)
        detmc = mc[:,0,0]*mc[:,1,1]-mc[:,0,1]*mc[:,1,0]
        detmc = detmc.reshape(-1,1)
        detm = m[:,0,0]*m[:,1,1]-m[:,0,1]*m[:,1,0]

        idx0 = cell[:, 0]
        idx1 = cell[:, 1] 
        idx2 = cell[:, 2] 

        v0 = node[idx2] - node[idx1]
        v1 = node[idx0] - node[idx2]
        v2 = node[idx1] - node[idx0]
        
        m_v0 = (m[idx2]+m[idx1])/2
        m_v1 = (m[idx0]+m[idx2])/2
        m_v2 = (m[idx1]+m[idx0])/2
        area = bm.sqrt(detmc)*0.5*(-v2[:, [0]]*v1[:, [1]] + v2[:, [1]]*v1[:, [0]])
        l2 = bm.zeros((NC, 3), dtype=bm.float64)

        l2[:,0] = mc[:,0,0]*v0[:,0]**2+mc[:,1,1]*v0[:,1]**2+2*mc[:,0,1]*v0[:,0]*v0[:,1]
        l2[:,1] = mc[:,0,0]*v1[:,0]**2+mc[:,1,1]*v1[:,1]**2+2*mc[:,0,1]*v1[:,0]*v1[:,1] 
        l2[:,2] = mc[:,0,0]*v2[:,0]**2+mc[:,1,1]*v2[:,1]**2+2*mc[:,0,1]*v2[:,0]*v2[:,1] 

        l = bm.sqrt(l2)
        p = l.sum(axis=1, keepdims=True)
        q = l.prod(axis=1, keepdims=True)
        quality = p*q/(16*area**2)
        ideal_area = bm.sum(area)/NC
        size_energy = 0.5*(area/ideal_area-1)**2
        penalty = self.penalty(area)
        quality = quality + penalty#+size_energy
        A0,A1,A2,B,C = self.grad_matrix(node=node)
        ttt = area/ideal_area-1
        gradp = bm.full_like(node,0.0)
        gradp[:,0] = (A0@node[:,0]+(A2+B)@node[:,1])
        gradp[:,1] = ((A2+B.T)@node[:,0]+A1@node[:,1])

        gradp = gradp[~self.isFixNode]
        gradp = self.tangent_project(gradp,x)
        gradp = gradp.T.flatten()
        return bm.mean(quality), gradp

    def penalty(self,q):
        pen = bm.zeros_like(q)
        flag = q<0
        pen[flag] = (1-q[flag]/1e-12)**2
        return pen

    def grad_matrix(self, node=None):
        NC = self.mesh.number_of_cells()
        NN = self.mesh.number_of_nodes()
        if node is None:
            node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')

        idx0 = cell[:, 0]
        idx1 = cell[:, 1]
        idx2 = cell[:, 2]

        m = self.metric(node)
        mc = (m[cell[:,0]] + m[cell[:,1]] + m[cell[:,2]])/3
        detmc = mc[:,0,0]*mc[:,1,1]-mc[:,0,1]*mc[:,1,0]
        detm = m[:,0,0]*m[:,1,1]-m[:,0,1]*m[:,1,0]
        detmc = detmc.reshape(-1,1)
        
        v0 = node[idx2] - node[idx1]
        v1 = node[idx0] - node[idx2]
        v2 = node[idx1] - node[idx0]

        m_v0 = (m[idx2]+m[idx1])/2
        m_v1 = (m[idx0]+m[idx2])/2
        m_v2 = (m[idx1]+m[idx0])/2
        area = 0.5*(-v2[:, [0]]*v1[:, [1]] + v2[:, [1]]*v1[:,[0]])*bm.sqrt(detmc)
        l2 = bm.zeros((NC, 3), dtype=bm.float64)

        l2[:,0] = mc[:,0,0]*v0[:,0]**2+mc[:,1,1]*v0[:,1]**2+2*mc[:,0,1]*v0[:,0]*v0[:,1]
        l2[:,1] = mc[:,0,0]*v1[:,0]**2+mc[:,1,1]*v1[:,1]**2+2*mc[:,0,1]*v1[:,0]*v1[:,1] 
        l2[:,2] = mc[:,0,0]*v2[:,0]**2+mc[:,1,1]*v2[:,1]**2+2*mc[:,0,1]*v2[:,0]*v2[:,1] 

        l = bm.sqrt(l2)
        p = l.sum(axis=1, keepdims=True)
        q = l.prod(axis=1, keepdims=True)
        mu = p*q/(16*area**2)

        c = mu*(1/(p*l) + 1/l2)/NC
        val0 = bm.concatenate((
            mc[:,0,0]*c[:, [1, 2]].sum(axis=1), -mc[:,0,0]*c[:, 2],-mc[:,0,0]*c[:, 1],
            -mc[:,0,0]*c[:, 2], mc[:,0,0]*c[:, [0, 2]].sum(axis=1), -mc[:,0,0]*c[:, 0],
            -mc[:,0,0]*c[:, 1], -mc[:,0,0]*c[:, 0], mc[:,0,0]*c[:, [0, 1]].sum(axis=1)))

        val1 = bm.concatenate((
            mc[:,1,1]*c[:, [1, 2]].sum(axis=1), -mc[:,1,1]*c[:, 2], -mc[:,1,1]*c[:, 1],
            -mc[:,1,1]*c[:, 2], mc[:,1,1]*c[:, [0, 2]].sum(axis=1), -mc[:,1,1]*c[:, 0],
            -mc[:,1,1]*c[:, 1], -mc[:,1,1]*c[:, 0], mc[:,1,1]*c[:, [0, 1]].sum(axis=1)))

        val2 = bm.concatenate((
            mc[:,0,1]*c[:, [1, 2]].sum(axis=1), -mc[:,0,1]*c[:, 2], -mc[:,0,1]*c[:, 1],
            -mc[:,0,1]*c[:, 2], mc[:,0,1]*c[:, [0, 2]].sum(axis=1), -mc[:,0,1]*c[:, 0],
            -mc[:,0,1]*c[:, 1], -mc[:,0,1]*c[:, 0], mc[:,0,1]*c[:, [0, 1]].sum(axis=1)))
        I = bm.concatenate((
            idx0, idx0, idx0,
            idx1, idx1, idx1,
            idx2, idx2, idx2))
        J = bm.concatenate((idx0, idx1, idx2))
        J = bm.concatenate((J, J, J))
        indices = bm.stack([I,J],axis=0)
        A0 = COOTensor(indices, val0, spshape=(NN, NN))
        A1 = COOTensor(indices, val1, spshape=(NN, NN))
        A2 = COOTensor(indices, val2, spshape=(NN, NN))
        cn = bm.sqrt(detmc)*mu/(area*NC)
        ideal_area = bm.sum(area)/NC
        cn2 = -0.5*ideal_area*bm.sqrt(detmc)/NC
        cn.shape = (cn.shape[0],)
        cn2.shape = (cn2.shape[0],)
        val = bm.concatenate((-cn, cn, cn, -cn, -cn, cn))
        val3 = bm.concatenate((-cn2, cn2, cn2, -cn2, -cn2, cn2))
        I = bm.concatenate((idx0, idx0, idx1, idx1, idx2, idx2))
        J = bm.concatenate((idx1, idx2, idx0, idx2, idx0, idx1))
        indices = bm.stack([I,J],axis=0)
        B = COOTensor(indices, val, spshape=(NN, NN))
        C = COOTensor(indices, val3, spshape=(NN, NN))
        return (A0,A1,A2,B,C)

    def update_m(self,x):
        GD = self.mesh.geo_dimension()
        node0 = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        node = bm.zeros_like(node0)
        node[self.isFixNode] = self.FixNode

        NN = self.mesh.number_of_nodes()
        NC = self.mesh.number_of_cells()

        n = len(x)//2
        node[self.isFreeNode,0] = x[:n]
        node[self.isFreeNode,1] = x[n:]

        NC = self.mesh.number_of_cells()
        NN = self.mesh.number_of_nodes()
        self.m = self.metric(node)
        self.mc = self.cell_metric(self.m)

    def flipopt(self):
        node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        NN = node.shape[0]
        newcell2,oldcelltag2 = self._flip22()
        Ne = oldcelltag2.shape[0]
        if Ne==0:
            return False
        flip_cells = []
        flip_cells += [tuple(row) for row in oldcelltag2]
        used = set()
        keep = []
        for cells in flip_cells:
            if any(c in used for c in cells):
                keep.append(False)
            else:
                keep.append(True)
                used.update(cells)
        keep = bm.array(keep)
        oldcelltag2 = oldcelltag2[keep]
        newcell2 = newcell2[bm.repeat(keep, 2)]
        cell[oldcelltag2[:,0]] = newcell2[::2]
        cell[oldcelltag2[:,1]] = newcell2[1::2]
        self.mesh.cell = cell
        self.mesh.construct()
        return True 

    def _flip22(self):
        node = self.mesh.entity('node')
        edge = self.mesh.entity('edge')
        cell = self.mesh.entity('cell')
        edge2cell = self.mesh.edge_to_cell()
        NE = edge.shape[0]
        edgeable = edge2cell[:,0] !=edge2cell[:,1]
        c0 = edge2cell[edgeable,0]
        c1 = edge2cell[edgeable,1]
        le0 = edge2cell[edgeable,2]
        le1 = edge2cell[edgeable,3]
        e0 = edge[edgeable]
        a = e0[:,0]
        b = e0[:,1]
        opp0 = cell[c0,le0]
        opp1 = cell[c1,le1]
        oldcell2 = bm.zeros((2*len(c0),3),dtype=cell.dtype)
        oldcell2[::2] = cell[c0]
        oldcell2[1::2] = cell[c1]

        newcell2 = bm.zeros((2 * len(c0), 3), dtype=cell.dtype)
        newcell2[::2, 0] = opp0
        newcell2[::2, 1] = opp1
        newcell2[::2, 2] = a

        newcell2[1::2, 0] = opp1
        newcell2[1::2, 1] = opp0
        newcell2[1::2, 2] = b
        
        pa = node[a]
        pb = node[b]
        pc = node[opp0]
        pd = node[opp1]
        def cross2(u, v):
            return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        s0 = cross2(pb - pa, pc - pa)
        s1 = cross2(pb - pa, pd - pa)
        convex = (s0 * s1) < -1e-14
        p0 = node[newcell2[:, 0]]
        p1 = node[newcell2[:, 1]]
        p2 = node[newcell2[:, 2]]

        area2 = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - \
                (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])


        neg = area2 < 0
        tmp = newcell2[neg, 1].copy()
        newcell2[neg, 1] = newcell2[neg, 2]
        newcell2[neg, 2] = tmp

        valid = convex

        c0 = c0[valid]
        c1 = c1[valid]
        oldcell2 = oldcell2[bm.repeat(valid, 2)]
        newcell2 = newcell2[bm.repeat(valid, 2)]

        q1 = self._quality(oldcell2)   # (2*N,)
        q2 = self._quality(newcell2)   # (2*N,)

        q1mean = bm.maximum(q1[::2], q1[1::2])
        q2mean = bm.maximum(q2[::2], q2[1::2])
        
        accept = q2mean < q1mean
        accept = accept.reshape(-1)
        if accept.sum() == 0:
            return bm.array([]), bm.array([])
        newcell2 = newcell2[bm.repeat(accept, 2)]
        c0 = c0[accept]
        c1 = c1[accept]
        return newcell2, bm.array([c0,c1]).T

    def _quality(self, cell):
        node = self.mesh.entity('node')
        NC = cell.shape[0] 
        GD =self.mesh.geo_dimension()
        m = self.metric(node)
        idxi = cell[:, 0]
        idxj = cell[:, 1] 
        idxk = cell[:, 2]
        Mi = m[idxi]
        Mj = m[idxj]
        Mk = m[idxk]
        mc = (Mi+Mj+Mk)/3

        detmc = mc[:,0,0]*mc[:,1,1]-mc[:,0,1]*mc[:,1,0]
        detmc = detmc.reshape(-1,1)
        detm = m[:,0,0]*m[:,1,1]-m[:,0,1]*m[:,1,0]

        idx0 = cell[:, 0]
        idx1 = cell[:, 1] 
        idx2 = cell[:, 2] 

        v0 = node[idx2] - node[idx1]
        v1 = node[idx0] - node[idx2]
        v2 = node[idx1] - node[idx0]
        
        m_v0 = (m[idx2]+m[idx1])/2
        m_v1 = (m[idx0]+m[idx2])/2
        m_v2 = (m[idx1]+m[idx0])/2
        
        area = bm.sqrt(detmc)*0.5*(-v2[:, [0]]*v1[:, [1]] + v2[:, [1]]*v1[:, [0]])
        l2 = bm.zeros((NC, 3), dtype=bm.float64)

        l2[:,0] = mc[:,0,0]*v0[:,0]**2+mc[:,1,1]*v0[:,1]**2+2*mc[:,0,1]*v0[:,0]*v0[:,1]
        l2[:,1] = mc[:,0,0]*v1[:,0]**2+mc[:,1,1]*v1[:,1]**2+2*mc[:,0,1]*v1[:,0]*v1[:,1] 
        l2[:,2] = mc[:,0,0]*v2[:,0]**2+mc[:,1,1]*v2[:,1]**2+2*mc[:,0,1]*v2[:,0]*v2[:,1] 

        l = bm.sqrt(l2)
        p = l.sum(axis=1, keepdims=True)
        q = l.prod(axis=1, keepdims=True)
        quality = p*q/(16*area**2)
        penalty = self.penalty(area)
        quality = quality + penalty
        return quality

    @staticmethod
    def get_quality(mesh,metric):
        GD = mesh.geo_dimension()
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()

        idx0 = cell[:, 0]
        idx1 = cell[:, 1] 
        idx2 = cell[:, 2] 

        v0 = node[idx2] - node[idx1]
        v1 = node[idx0] - node[idx2]
        v2 = node[idx1] - node[idx0]

        m = metric(node)
        mc = (m[cell[:,0]] + m[cell[:,1]] + m[cell[:,2]])/3
        detmc = mc[:,0,0]*mc[:,1,1]-mc[:,0,1]*mc[:,1,0]
        detmc = detmc.reshape(-1,1)
        detm = m[:,0,0]*m[:,1,1]-m[:,0,1]*m[:,1,0]

        m_v0 = (m[idx2]+m[idx1])/2
        m_v1 = (m[idx0]+m[idx2])/2
        m_v2 = (m[idx1]+m[idx0])/2

        area = 0.5*(-v2[:, [0]]*v1[:, [1]] + v2[:, [1]]*v1[:,[0]])*bm.sqrt(detmc)
        l2 = bm.zeros((NC, 3), dtype=bm.float64)
        
        l2[:,0] = mc[:,0,0]*v0[:,0]**2+mc[:,1,1]*v0[:,1]**2+2*mc[:,0,1]*v0[:,0]*v0[:,1]
        l2[:,1] = mc[:,0,0]*v1[:,0]**2+mc[:,1,1]*v1[:,1]**2+2*mc[:,0,1]*v1[:,0]*v1[:,1] 
        l2[:,2] = mc[:,0,0]*v2[:,0]**2+mc[:,1,1]*v2[:,1]**2+2*mc[:,0,1]*v2[:,0]*v2[:,1] 
       
        l = bm.sqrt(l2)
        p = l.sum(axis=1, keepdims=True)
        q = l.prod(axis=1, keepdims=True)
        quality = (16*area**2)/(p*q)
        return quality
