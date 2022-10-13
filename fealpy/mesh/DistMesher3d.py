import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

from .TetrahedronMesh import TetrahedronMesh 

class DistMesher3d():

    def __init__(self,
            domain, 
            hmin,
            ptol = 0.001,
            ttol = 0.01,
            fscale = 1.1,
            dt = 0.05,
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

        self.localEdge = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
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
        zh = box[5] - box[4]
        nx = int(xh/hmin) + 1
        ny = int(yh/(hmin*np.sqrt(3)/2)) + 1
        nz = int(zh/(hmin*np.sqrt(2/3))) + 1 

        NN = (nx+1)*(ny+1)*(nz+1)
        node = np.zeros((NN, 3), dtype=np.float64)
        X, Y, Z = np.mgrid[
                box[0]:box[1]:complex(0, nx+1), 
                box[2]:box[3]:complex(0, ny+1),
                box[4]:box[5]:complex(0, nz+1)
                ]

        X[:, 1::2, :] += hmin/2
        Y[:, :, 0::2] += hmin*np.sqrt(3)/3


        node[:, 0] = X.flatten()
        node[:, 1] = Y.flatten()
        node[:, 2] = Z.flatten()

        node = node[fd(node) < self.geps, :]

        r0 = fh(node)**3
        val = np.min(r0)/r0
        NN = len(node)
        node = node[np.random.random(NN) < val]


        fnode = self.domain.facet(0) # 区域中的固定点
        if fnode is not None:
            #TODO: fnode 和 node 中可能存在重复点
            node = np.concatenate((fnode, node), axis=0)

        return node


    def delaunay(self, node):
        fd = self.domain.signed_dist_function
        # 其中 Qz 是增加一个无穷远点
        tet = Delaunay(node, qhull_options='Qt Qbb Qc Qz')
        cell = np.asarray(tet.simplices, dtype=np.int_)
        bc = (node[cell[:, 0]] + node[cell[:, 1]] + node[cell[:, 2]] +
                node[cell[:, 3]])/4
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
            mesh = TetrahedronMesh(node, cell)
            bc = mesh.entity_barycenter('cell')
            flag = bc[:, 0] < 0.0 
            mesh.celldata['flag'] = flag 
            mesh.to_vtk(fname=fname)

        return edge

    def projection(self, node, d):
        """
        @brief 把移动到区域外面的点投影到边界上

        @param[in] node 移动到区域外面的点
        @param[in] 
        """

        domain = self.domain
        fd = domain.signed_dist_function

        if hasattr(domain, 'projection'):
            node = domain.projection(node)
        else:
            depsx = np.array([self.deps, 0, 0])
            depsy = np.array([0, self.deps, 0])
            depsz = np.array([0, 0, self.deps])
            dgradx = (fd(node + depsx) - d)/self.deps
            dgrady = (fd(node + depsy) - d)/self.deps
            dgradz = (fd(node + depsz) - d)/self.deps
            node[:, 0] = node[:, 0] - d*dgradx
            node[:, 1] = node[:, 1] - d*dgrady
            node[:, 2] = node[:, 2] - d*dgradz

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
        L0 = np.power(np.sum(L**3)/np.sum(he**3), 1/3)*self.fscale*he
        F = np.maximum(L0 - L, 0)
        FV = (F/L)[:, None]*v

        dnode = np.zeros(node.shape, dtype=np.float64)

        np.add.at(dnode[:, 0], edge[:, 0], FV[:, 0])
        np.add.at(dnode[:, 1], edge[:, 0], FV[:, 1])
        np.add.at(dnode[:, 2], edge[:, 0], FV[:, 2])
        np.subtract.at(dnode[:, 0], edge[:, 1], FV[:, 0])
        np.subtract.at(dnode[:, 1], edge[:, 1], FV[:, 1])
        np.subtract.at(dnode[:, 2], edge[:, 1], FV[:, 2])

        fnode = self.domain.facet(0)
        if fnode is not None:
            n = len(fnode)
            dnode[0:n, :] = 0.0

        node += self.dt*dnode
        md = np.sqrt(np.sum(dnode**2, axis=1))

        return md 

    def post_processing(self, node):
        """
        """
        domain = self.domain
        fd = domain.signed_dist_function
        fh = domain.sizing_function 
        deps = self.deps

        cell = self.delaunay(node)
        mesh = TetrahedronMesh(node, cell)
        face2cell = mesh.ds.face_to_cell()
        isBdFace = (face2cell[:, 0] == face2cell[:, 1])
        cidx = face2cell[isBdFace, 0]
        lidx = face2cell[isBdFace, 2]
        nidx = cell[cidx, lidx]

        c = mesh.circumcenter(index=cidx)
        d = fd(c)
        isOut = (d > -deps)
        idx = nidx[isOut]
        if len(idx) > 0:
            p0 = node[idx]
            if hasattr(domain, 'projection'):
                node[idx] = domain.projection(node[idx])
            else:
                depsx = np.array([self.deps, 0, 0])
                depsy = np.array([0, self.deps, 0])
                depsz = np.array([0, 0, self.deps])
                dgradx = (fd(node + depsx) - d)/self.deps
                dgrady = (fd(node + depsy) - d)/self.deps
                dgradz = (fd(node + depsz) - d)/self.deps
                node[:, 0] = node[:, 0] - d*dgradx
                node[:, 1] = node[:, 1] - d*dgrady
                node[:, 2] = node[:, 2] - d*dgradz

    def meshing(self, maxit=1000):
        """
        """

        domain = self.domain
        fd = domain.signed_dist_function

        if hasattr(domain, 'init_nodes'):
            node = domain.init_nodes(self.geps)
        else:
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
                print("第 %05d 次三角化"%(self.NT))

            md = self.move(node, edge)

            d = fd(node)
            isOut = d > -self.deps 
            if np.any(isOut):
                node[isOut] = self.projection(node[isOut], d[isOut])

            if self.dt*np.max(md) < self.ptol*self.hmin:
                break
            else:
                mmove = np.max(np.sqrt(np.sum((node - p0)**2, axis=1)))
                p0[:] = node

        cell = self.delaunay(node)
        mesh = TetrahedronMesh(node, cell)

        # 把边界点投影到边界上
        isBdNode = mesh.ds.boundary_node_flag()
        fnode = self.domain.facet(0)
        if fnode is not None:
            n = len(fnode)
            isBdNode[0:n] = False

        depsx = np.array([self.deps, 0, 0])
        depsy = np.array([0, self.deps, 0])
        depsz = np.array([0, 0, self.deps])
        for i in range(3):
            bnode = node[isBdNode]
            d = fd(bnode)
            dgradx = (fd(bnode + depsx) - d)/self.deps
            dgrady = (fd(bnode + depsy) - d)/self.deps
            dgradz = (fd(bnode + depsz) - d)/self.deps
            dgrad2 = dgradx**2 + dgrady**2 + dgradz**2
            dgradx /= dgrad2
            dgrady /= dgrad2
            dgradz /= dgrad2
            node[isBdNode, 0] = bnode[:, 0] - d*dgradx
            node[isBdNode, 1] = bnode[:, 1] - d*dgrady
            node[isBdNode, 2] = bnode[:, 2] - d*dgradz

        return mesh 



