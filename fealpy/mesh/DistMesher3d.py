import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

from .TetrahedronMesh import TetrahedronMesh 

class DistMesher3d():

    def __init__(self,
            domain, 
            hmin,
            ptol = 0.001,
            ttol = 0.1,
            fscale = 1.1,
            dt = 0.1):
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
        self.ttol = ttol # 单元删除阈值
        self.fscale = fscale

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
        ny = int(yh/hmin) + 1
        nz = int(zh/hmin) + 1 

        NN = (nx+1)*(ny+1)*(nz+1)
        node = np.zeros((NN, 3), dtype=np.float64)
        X, Y, Z = np.mgrid[
                box[0]:box[1]:complex(0, nx+1), 
                box[2]:box[3]:complex(0, ny+1),
                box[4]:box[5]:complex(0, nz+1)
                ]
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
        d = Delaunay(node)
        cell = np.asarray(d.simplices, dtype=np.int_)
        bc = (node[cell[:, 0]] + node[cell[:, 1]] + node[cell[:, 2]] +
                node[cell[:, 2]])/4
        return  cell[fd(bc) < -self.geps, :]

    def meshing(self, maxit=1000):
        """
        @brief 运行
        """
        fd = self.domain.signed_dist_function
        fh = self.domain.sizing_function 
        hmin = self.hmin
        dt = self.dt
        fscale = self.fscale

        localEdge = self.localEdge
        ptol = self.ptol
        ttol = self.ttol
        deps = self.deps
        geps = self.geps

        node = self.init_nodes()

        NT = 0
        mmove = 1e+10
        count = 0 
        while count < maxit: 
            count += 1
            if mmove > self.ttol*self.hmin:
                p0 = node.copy()
                cell = self.delaunay(node)
                NT += 1
                print("第 %05d 次三角化"%(NT))
                totalEdge = cell[:, localEdge].reshape(-1, 2)
                edge  = np.unique(np.sort(totalEdge, axis=1), axis=0)

                fname = "mesh-%05d.vtu"%(NT)
                mesh = TetrahedronMesh(node, cell)
                mesh.to_vtk(fname=fname)

            v = node[edge[:, 0]] - node[edge[:, 1]]
            L = np.sqrt(np.sum(v**2, axis=1))
            bc = (node[edge[:, 0]] + node[edge[:, 1]])/2.0
            he = fh(bc) 
            L0 = np.power(np.sum(L**3)/np.sum(he**3), 1/3)*fscale*he
            F = np.minimum(L0 - L, 0)
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

            node += dt*dnode

            d = fd(node)
            idx = d > 0
            depsx = np.array([self.deps, 0, 0])
            depsy = np.array([0, self.deps, 0])
            depsz = np.array([0, 0, self.deps])
            dgradx = (fd(node[idx, :] + depsx) - d[idx])/deps
            dgrady = (fd(node[idx, :] + depsy) - d[idx])/deps
            dgradz = (fd(node[idx, :] + depsz) - d[idx])/deps
            node[idx, 0] = node[idx, 0] - d[idx]*dgradx
            node[idx, 1] = node[idx, 1] - d[idx]*dgrady
            node[idx, 2] = node[idx, 2] - d[idx]*dgradz

            md = dt*np.max(np.sqrt(np.sum(dnode[d < -geps]**2, axis=1)))
            if md < ptol*hmin:
                break
            else:
                mmove = np.max(np.sqrt(np.sum((node - p0)**2, axis=1)))


        cell = self.delaunay(node)
        return TetrahedronMesh(node, cell)

    
