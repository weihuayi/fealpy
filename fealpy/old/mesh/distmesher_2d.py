import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

from .triangle_mesh import TriangleMesh 

class DistMesher2d():

    def __init__(self,
            domain, 
            hmin,
            ptol = 0.001,
            ttol = 0.01,
            fscale = 1.2,
            dt = 0.2,
            output=False):
        """
        @brief 

        @param[in] domain 三维区域
        @param[in] hmin 最小的边长
        @param[in] ptol
        @param[in] ttol
        @param[in] fscale
        @param[in] dt
        """

        self.localEdge = np.array([(0, 1), (1, 2), (2, 0)])
        self.domain = domain
        self.hmin = hmin
        self.ptol = ptol # 初始点拒绝阈值
        self.ttol = ttol # 重新三角化阈值
        self.fscale = fscale
        self.output = output

        eps = np.finfo(float).eps
        self.geps = 0.1*hmin
        self.deps = np.sqrt(eps)*hmin
        self.dt = dt 

        self.NT = 0 # 记录三角化的次数


    def init_nodes(self): 
        """
        @brief 生成初始网格
        """

        fd = self.domain.signed_dist_function
        fh = self.domain.sizing_function 
        box = self.domain.box
        hmin = self.hmin


        xh = box[1] - box[0]
        yh = box[3] - box[2]
        N = int(xh/hmin)+1
        M = int(yh/(hmin*np.sqrt(3)/2)) + 1

        mg = np.mgrid[box[2]:box[3]:complex(0, M), box[0]:box[1]:complex(0, N)]
        x = mg[1, :, :]
        y = mg[0, :, :]
        x[1::2, :] += hmin/2
        node = np.concatenate(
                (x.reshape(-1, 1), y.reshape((-1,1))), 
                axis=1)
        node = node[fd(node) < -self.geps, :]
        r0 = 1/fh(node)**2
        NN = len(node)
        node = node[np.random.random((NN, )) < r0/np.max(r0),:]

        fnode = self.domain.facet(0) # 区域中的固定点
        if fnode is not None:
            # TODO: 重复点的问题
            node = np.concatenate((fnode, node), axis=0)
        return node

    def delaunay(self, node):
        fd = self.domain.signed_dist_function
        tri = Delaunay(node, qhull_options='Qt Qbb Qc Qz')
        cell = np.asarray(tri.simplices, dtype=np.int_)
        bc = (node[cell[:, 0]] + node[cell[:, 1]] + node[cell[:, 2]])/3
        return  cell[fd(bc) < -self.geps]

    def construct_edge(self, node):
        """
        @brief 生成网格的边
        """
        localEdge = self.localEdge
        cell = self.delaunay(node)
        totalEdge = cell[:, localEdge].reshape(-1, 2)
        edge  = np.unique(np.sort(totalEdge, axis=1), axis=0)

        if self.output:
            fname = "mesh-%05d.vtu"%(self.NT)
            mesh = TriangleMesh(node, cell)
            bc = mesh.entity_barycenter('cell')
            flag = bc[:, 0] < 0.0 
            mesh.celldata['flag'] = flag 
            mesh.to_vtk(fname=fname)

        return edge

    def project(self, node, d):
        """
        @brief 把移动到区域外面的点投影到边界上

        @param[in] node 移动到区域外面的点
        @param[in] 
        """

        domain = self.domain
        fd = domain.signed_dist_function

        if hasattr(domain, 'project'):
            node = domain.project(node)
        else:
            depsx = np.array([self.deps, 0])
            depsy = np.array([0, self.deps])
            dgradx = (fd(node + depsx) - d)/self.deps
            dgrady = (fd(node + depsy) - d)/self.deps
            node[:, 0] = node[:, 0] - d*dgradx
            node[:, 1] = node[:, 1] - d*dgrady

        return node

    def move(self, node, edge):
        """
        @brief 移动节点
        
        @return md 每个节点移动的距离
        """

        fh = self.domain.sizing_function

        v = node[edge[:, 0]] - node[edge[:, 1]]
        L = np.sqrt(np.sum(v**2, axis=1))
        bc = (node[edge[:, 0]] + node[edge[:, 1]])/2.0
        he = fh(bc) 
        L0 = np.sqrt(np.sum(L**2)/np.sum(he**2))*self.fscale*he
        F = np.maximum(L0 - L, 0)
        FV = (F/L)[:, None]*v

        dnode = np.zeros(node.shape, dtype=np.float64)


        np.add.at(dnode[:, 0], edge[:, 0], FV[:, 0])
        np.add.at(dnode[:, 1], edge[:, 0], FV[:, 1])
        np.subtract.at(dnode[:, 0], edge[:, 1], FV[:, 0])
        np.subtract.at(dnode[:, 1], edge[:, 1], FV[:, 1])


        fnode = self.domain.facet(0)
        if fnode is not None:
            n = len(fnode)
            dnode[0:n, :] = 0.0

        node += self.dt*dnode
        md = np.sqrt(np.sum(dnode**2, axis=1))

        return md 

    def meshing(self, maxit=1000):
        """
        """

        domain = self.domain
        fd = domain.signed_dist_function

        node = self.init_nodes()
        p0 = node.copy()
        self.NT = 0
        mmove = 1e+10
        count = 0 
        while count < maxit:
            count += 1
            if mmove > self.ttol*self.hmin:
                edge = self.construct_edge(node)
                self.NT += 1

            md = self.move(node, edge)

            d = fd(node)
            isOut = d > 0
            if np.any(isOut):
                node[isOut] = self.project(node[isOut], d[isOut])
             
            if self.dt*np.max(md[~isOut]) < self.ptol*self.hmin:
                break
            else:
                mmove = np.max(np.sqrt(np.sum((node - p0)**2, axis=1)))
                p0[:] = node

        self.post_processing(node)

        cell = self.delaunay(node)
        mesh = TriangleMesh(node, cell)

        # 把边界点投影到边界上
        isBdNode = mesh.ds.boundary_node_flag()
        fnode = self.domain.facet(0)
        if fnode is not None:
            n = len(fnode)
            isBdNode[0:n] = False

        depsx = np.array([self.deps, 0])
        depsy = np.array([0, self.deps])
        for i in range(2):
            bnode = node[isBdNode]
            d = fd(bnode)
            dgradx = (fd(bnode + depsx) - d)/self.deps
            dgrady = (fd(bnode + depsy) - d)/self.deps
            dgrad2 = dgradx**2 + dgrady**2
            dgradx /= dgrad2
            dgrady /= dgrad2
            node[isBdNode, 0] = bnode[:, 0] - d*dgradx
            node[isBdNode, 1] = bnode[:, 1] - d*dgrady

        return mesh 


    def post_processing(self, node):
        """
        """
        ne = np.array([1, 2, 0])
        pr = np.array([2, 0, 1])
        domain = self.domain
        fd = domain.signed_dist_function
        fh = domain.sizing_function 
        deps = self.deps

        cell = self.delaunay(node)
        mesh = TriangleMesh(node, cell)
        edge2cell = mesh.ds.edge_to_cell()
        isBdEdge = (edge2cell[:, 0] == edge2cell[:, 1])
        cidx = edge2cell[isBdEdge, 0]
        lidx = edge2cell[isBdEdge, 2]
        nidx = cell[cidx, lidx]

        c = mesh.circumcenter(index=cidx)
        d = fd(c)
        isOut = (d > -deps)
        nidx = nidx[isOut]
        cidx = cidx[isOut]
        lidx = lidx[isOut]
        if len(nidx) > 0:
            node[nidx] = (node[cell[cidx, ne[lidx]]] + node[cell[cidx, pr[lidx]]])/2.0


