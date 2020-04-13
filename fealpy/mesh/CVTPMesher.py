import numpy as np
from scipy.spatial import KDTree


class CVTPMesher:
    def __init__(self, domain):
        """
        Parameters
        ----------
        domain : HalEdgeDomain
        """
        self.domain = domain

    def meshing(self, refine=0, c=0.618, theta=100):

        bnode, hedge2bnode = self.boundary_meshing(
                refine=refine, c=c, theta=theta)
        newNode = self.init_interior_nodes(bnode, hedge2bnode)
        

    def boundary_meshing(self, n=0, c=0.618, theta=100):
        self.domain.boundary_uniform_refine(n=n)
        NN = self.domain.NV
        node = self.domain.vertices
        halfedge = self.domain.halfedge
    
        # 这里假设所有边的尺寸是一样的
        # 进一步的算法改进中，这些尺寸应该是自适应的
        # 顶点处的半径应该要平均平均一下
        idx0 = halfedge[halfedge[:, 3], 0]
        idx1 = halfedge[:, 0]
        v = node[idx1] - node[idx0]
        h = np.sqrt(np.sum(v**2, axis=-1))
        r = np.zeros(NN, dtype=self.domain.ftype)
        n = np.zeros(NN, dtype=self.domain.itype)
        np.add.at(r, idx0, h)
        np.add.at(r, idx1, h)
        np.add.at(n, idx0, 1)
        np.add.at(n, idx1, 1)
        r /= n
        r *= c
        w = np.array([[0, 1], [-1, 0]])

        # 修正角点相邻点的半径， 如果角点的角度小于 theta 的
        # 这里假设角点相邻的节点， 到角点的距离相等
        isFixed = self.domain.fixed[halfedge[:, 0]]
        idx, = np.nonzero(isFixed)
        pre = halfedge[idx, 3]
        nex = halfedge[idx, 2]

        p0 = node[halfedge[pre, 0]]
        p1 = node[halfedge[idx, 0]]
        p2 = node[halfedge[nex, 0]]

        v0 = p2 - p1
        v1 = p0 - p1
        l0 = np.sqrt(np.sum(v0**2, axis=-1))
        l1 = np.sqrt(np.sum(v1**2, axis=-1))
        s = np.cross(v0, v1)/l0/l1
        c = np.sum(v0*v1, axis=-1)/l0/l1
        a = np.arcsin(s)
        a[s < 0] += 2*np.pi
        a[c == -1] = np.pi
        a = np.degrees(a)
        isCorner = a < theta
        idx = idx[isCorner] # 需要特殊处理的半边编号 


        v2 = (v0[isCorner] + v1[isCorner])/2
        v2 /= np.sqrt(np.sum(v2**2, axis=-1, keepdims=True))
        v2 *= r[halfedge[idx, 0], None] 
        p = node[halfedge[idx, 0]] + v2
        r[halfedge[pre[isCorner], 0]] = np.sqrt(np.sum((p - p0[isCorner])**2, axis=-1))
        r[halfedge[nex[isCorner], 0]] = np.sqrt(np.sum((p - p2[isCorner])**2, axis=-1))

        # 把一些生成的点合并掉, 这里只检查当前半边和下一个半边的生成的点
        # 这里也假设很近的点对是孤立的. 
        NG = halfedge.shape[0] # 会生成 NG 个生成子, 每个半边都对应一个
        index = np.arange(NG)
        nex = halfedge[idx, 2]
        index[nex] = idx
        
        # 计算每个半边对应的节点
        center = (node[idx0] + node[idx1])/2
        r0 = r[idx0]
        r1 = r[idx1]
        c0 = 0.5*(r0**2 - r1**2)/h**2
        c1 = 0.5*np.sqrt(2*(r0**2 + r1**2)/h**2 - (r0**2 - r1**2)**2/h**4 - 1)
        bnode = center + c0.reshape(-1, 1)*v + c1.reshape(-1, 1)*(v@w) 

        isKeepNode = np.zeros(NG, dtype=np.bool)
        isKeepNode[index] = True
        idxmap = np.zeros(NG, dtype=np.int)
        idxmap[isKeepNode] = range(isKeepNode.sum())

        bnode = bnode[isKeepNode]
        pf = node[halfedge[idx, 0]] -v2
        bnode = np.append(bnode,pf,axis=0)
        hedge2bnode = idxmap[index]
        return bnode, hedge2bnode


    def init_interior_nodes(self, bnode, hedge2bnode):
        
        node = self.domain.vertices
        halfedge = self.domain.halfedge

        NB = bnode.shape[0]
        bnode2subdomain = np.zeros(NB, dtype=np.int)
        bnode2subdomain[hedge2bnode] = halfedge[:, 1]

        idx0 = halfedge[halfedge[:, 3], 0]
        idx1 = halfedge[:, 0]
        v = node[idx1] - node[idx0]
        h = np.sqrt(np.sum(v**2, axis=-1))

        tree = KDTree(bnode)
        c = 6*np.sqrt(3*(h[0]/2)*(h[0]/4)**3/2)

        for index in filter(lambda x: x > 0, self.domain.subdomain):
            p = bnode[bnode2subdomain == index]
            xmin = min(p[:, 0])
            xmax = max(p[:, 0])
            ymin = min(p[:, 1])
            ymax = max(p[:, 1])
            
            area = self.domain.area[index]
            N = int(area/c)
            N0 = p.shape[0]
            start = 0
            newNode = np.zeros((N - N0, 2), dtype=node.dtype)
            NN = newNode.shape[0]
            while True:
                pp = np.random.rand(NN-start, 2)
                pp *= np.array([xmax-xmin,ymax-ymin])
                pp += np.array([xmin,ymin])
                d, idx = tree.query(pp)
                flag0 = d > (h[0]/2)
                flag1 = (bnode2subdomain[idx] == index)
                pp = pp[flag0 & flag1]# 筛选出符合要求的点
                end = start + pp.shape[0]
                newNode[start:end] = pp
                if end == NN:
                    break
                else:
                    start = end

        return newNode 
    

    def voronoi(self, node):
        pass
