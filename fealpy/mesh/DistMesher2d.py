import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

from .TriangleMesh import TriangleMesh 

class DistMesher2d():

    def __init__(self,
            domain, 
            hmin,
            ptol = 0.001,
            ttol = 0.1,
            fscale = 1.2,
            dt = 0.2,
            output=True):
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
        self.geps = 0.001*hmin
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
        d = Delaunay(node)
        cell = np.asarray(d.simplices, dtype=np.int_)
        bc = (node[cell[:, 0]] + node[cell[:, 1]] + node[cell[:, 2]] +
                node[cell[:, 2]])/4
        return  cell[fd(bc) < -self.geps]

    def meshing(self, maxit=1000):
        """
        @brief 运行
        """
        domain = self.domain
        fd = domain.signed_dist_function
        fh = domain.sizing_function 
        hmin = self.hmin
        dt = self.dt
        fscale = self.fscale
        output = self.output

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
                print("第 %05d 次三角化"%(NT))
                p0 = node.copy()
                cell = self.delaunay(node)
                NT += 1
                totalEdge = cell[:, localEdge].reshape(-1, 2)
                edge  = np.unique(np.sort(totalEdge, axis=1), axis=0)

                
                if output:
                    fname = "mesh-%05d.vtu"%(NT)
                    mesh = TriangleMesh(node, cell)
                    bc = mesh.entity_barycenter('cell')
                    d = bc[:, 0] < 0.0 
                    mesh.celldata['dist'] = d
                    mesh.to_vtk(fname=fname)

            v = node[edge[:, 0]] - node[edge[:, 1]]
            L = np.sqrt(np.sum(v**2, axis=1))
            bc = (node[edge[:, 0]] + node[edge[:, 1]])/2.0
            he = fh(bc) 
            L0 = np.sqrt(np.sum(L**2)/np.sum(he**2))*fscale*he
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

            node += dt*dnode

            d = fd(node)
            idx = d > 0

            if hasattr(domain, 'projection'):
                node[idx] = domain.projection(node[idx])
            else:
                depsx = np.array([self.deps, 0])
                depsy = np.array([0, self.deps])
                dgradx = (fd(node[idx, :] + depsx) - d[idx])/deps
                dgrady = (fd(node[idx, :] + depsy) - d[idx])/deps
                node[idx, 0] = node[idx, 0] - d[idx]*dgradx
                node[idx, 1] = node[idx, 1] - d[idx]*dgrady

            md = dt*np.max(np.sqrt(np.sum(dnode[~idx]**2, axis=1)))
            if md < ptol*hmin:
                break
            else:
                mmove = np.max(np.sqrt(np.sum((node - p0)**2, axis=1)))

        cell = self.delaunay(node)
        return TriangleMesh(node, cell)

    
