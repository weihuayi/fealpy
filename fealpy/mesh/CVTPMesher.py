import numpy as np
from scipy.spatial import KDTree
import pdb
from scipy.spatial import Voronoi
from .PolygonMesh import PolygonMesh

class CVTPMesher:
    def __init__(self, mesh,dof = None):
        """
        Parameters
        ----------
        Mesh : 利用HalfEdgeMesh2d生成的网格
        dof : 输入是否为角点,True为角点, False为非角点
        """
        self.mesh = mesh
        if dof is not None:
            self.dof = dof
        else :
            self.dof = np.ones(len(mesh.node))

    def voronoi_meshing(self, nb=2, c=0.618, theta=100,adaptive = False,times = None):
        """
        Parameters
        ----------
        nb : 边界一致加密次数
        c : 参数, 控制voroCrust算法边界处圆的半径
        theta : 对角度小于theta的角点处的角进行角点处理
        times : 数组, 若不采用一致加密, 则输入times参数输入每个边的加密次数
        ----------
        该函数利用voroCrust算法确定边界重构所需要的种子点, 之后在内部撒入点, 并
        生成初始voronoi网格
        """
        self.vorocrust_boundary_matching(n=nb, c=c, theta=theta,adaptive =
                adaptive,times = times)
        self.init_interior_nodes()
        vor = self.voronoi()
        return vor

    def boundary_uniform_refine(self, n=2, times = None):
        """
        Parameters
        ----------
        n : 边界一致加密次数
        times :数组, 若不采用一致加密, 则输入times参数输入每个边的加密次数
        ----------
        该函数用于对输入的网格边界进行加密
        """
        mesh = self.mesh
        for i in range(n):
            node = mesh.node
            halfedge = mesh.ds.halfedge
            dof = self.dof
            isMarkedHEdge = mesh.ds.main_halfedge_flag()
            idx = halfedge[isMarkedHEdge,4]
            ec = (node[halfedge[isMarkedHEdge,0]]+node[halfedge[idx,0]])/2
            isMarkedHEdge[idx] = True

            mesh.init_level_info()
            mesh.refine_halfedge(isMarkedHEdge)
            self.dof = np.r_['0',
                    dof,np.zeros_like(ec[:, 0], dtype=np.bool)]

    def boundary_adaptive_refine(self, n = 2, times = None):
        """
        parameters
        ----------
        n : 最长边加密次数
        times : 支持手动输入各边加密次数
        ----------
        边界自适应加密
        """
        mesh = self.mesh
        halfedge = mesh.ds.halfedge
        node = mesh.node
        if times is None:
            idx0 = halfedge[halfedge[:, 3], 0]
            idx1 = halfedge[:, 0]
            v = node[idx1] - node[idx0]
            h = np.sqrt(np.sum(v**2, axis=-1))
            minh = min(h)
            maxh = max(h)
            if min(h)/max(h)>0.9:
                self.boundary_uniform_refine(n=n)
                return
            times = np.zeros(len(h))
            uqh = np.unique(h)
            t, = np.where(uqh == maxh)
            uqh = np.delete(uqh,t)
            times[h==maxh] = n
            mh = maxh/2**n
            for i in uqh:
                i0 = i/mh
                j = 0
                while(i0>1):
                    i0 /=2
                    j += 1
                times[h==i] = j
            times = times[::2]
        unique = np.unique(times)
        for i in unique:
            l = len(halfedge)
            isMarkedHEdge = np.zeros(l,dtype = np.bool_)
            isMarkedHEdge[::2] = (times == i)
            for j in range(int(i)):
                dof = self.dof
                idx = halfedge[isMarkedHEdge,4]
                ec = (node[halfedge[isMarkedHEdge,0]]+node[halfedge[idx,0]])/2
                isMarkedHEdge[idx] = True

                mesh.init_level_info()
                mesh.refine_halfedge(isMarkedHEdge)
                self.dof = np.r_['0',
                        dof,np.zeros_like(ec[:, 0], dtype=np.bool)]
                halfedge = mesh.ds.halfedge
                times = np.hstack((times,i*np.ones(int((len(halfedge)-l)/2))))
                l = len(halfedge)
                isMarkedHEdge = np.zeros(l,dtype = np.bool_)
                isMarkedHEdge[::2] = (times == i)

    def vorocrust_boundary_matching(self, n=0, c=0.618, theta=100,adaptive =
            False,times=None):
        """
        Parameters
        ----------
        n : 边界一致加密次数
        c : 参数, 控制voroCrust算法边界处圆的半径
        theta : 对角度小于theta的角点处的角进行角点处理
        times : 数组, 若不采用一致加密, 则输入times参数输入每个边的加密次数
        ----------
        该函数对边界进行加密并利用voroCrust算法确定重构边界所需要的种子点
        """
        if adaptive == False:
            self.boundary_uniform_refine(n=n)
        else :
            self.boundary_adaptive_refine(n=n,times=times)
        node = self.mesh.node
        sdomain = self.mesh.ds.subdomain
        NN = len(node)
        halfedge = self.mesh.entity('halfedge')
        #halfedge = self.mesh.ds.halfedge
    
        # 这里假设所有边的尺寸是一样的
        # 进一步的算法改进中, 这些尺寸应该是自适应的
        # 顶点处的半径应该要平均一下

        idx0 = halfedge[halfedge[:, 3], 0]
        idx1 = halfedge[:, 0]
        v = node[idx1] - node[idx0]
        h = np.sqrt(np.sum(v**2, axis=-1))
        r = np.zeros(NN, dtype=self.mesh.ftype)
        n = np.zeros(NN, dtype=self.mesh.itype)
        np.add.at(r, idx0, h)
        np.add.at(r, idx1, h)
        np.add.at(n, idx0, 1)
        np.add.at(n, idx1, 1)
        r /= n
        r *= c # 半径
        w = np.array([[0, 1], [-1, 0]])

        # 修正角点相邻点的半径, 如果角点的角度小于 theta 的
        # 这里假设角点相邻的节点, 到角点的距离相等
        if max(sdomain) == 1:
            isFixed = self.dof
            hnode = np.zeros(NN,dtype = np.int_)
            hnode[halfedge[:,0]] = np.arange(len(halfedge))
            hnode1 = halfedge[halfedge[hnode,2],4]

            idx, = np.where(isFixed==1)
            idx = np.hstack((hnode[idx],hnode1[idx]))
            pre = halfedge[idx, 3]
            nex = halfedge[idx, 2]
        else:
            isFixed, = np.where(self.dof==1)
            idx = np.zeros(len(halfedge),dtype = np.bool_)
            for i in isFixed:
                idx[halfedge[:,0]==i] = True
            idx, = np.where(idx == 1)
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
        aflag1 = ((c<0) & (a>np.pi))
        a[aflag1] = 3*np.pi - a[aflag1]
        aflag2 = ((c<0) & (a<(np.pi/2)))
        a[aflag2] = np.pi - a[aflag2]
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
        index[nex] = idx #半边对应的bnode 

        # 计算每个半边对应的节点
        center = (node[idx0] + node[idx1])/2
        r0 = r[idx0]
        r1 = r[idx1]
        c0 = 0.5*(r0**2 - r1**2)/h**2
        c1 = 0.5*np.sqrt(2*(r0**2 + r1**2)/h**2 - (r0**2 - r1**2)**2/h**4 - 1)
        bnode = center + c0.reshape(-1, 1)*v + c1.reshape(-1, 1)*(v@w)

        isKeepNode = np.zeros(NG, dtype=np.bool_)
        isKeepNode[index] = True
        idxmap = np.zeros(NG, dtype=np.int)
        idxmap[isKeepNode] = range(isKeepNode.sum())

        self.bnode = bnode[isKeepNode] #
        idxflag = halfedge[idx,1]<=0
        self.cnode = node[halfedge[idx[idxflag], 0]] - v2[idxflag] #
        self.hedge2bnode = idxmap[index] # hedge2bnode[i]: the index of node in bnode
        self.chedge = idx # the index of halfedge point on corner point

    def init_interior_nodes(self):
        """
        该函数对进行了边界重构的网格撒入内部点
        """
        mesh = self.mesh
        node = self.mesh.node
        halfedge = self.mesh.entity('halfedge')

        NNB = len(self.bnode)
        NNC = len(self.cnode)
        hcell = len(mesh.ds.hcell)# hcell[i] 是第i个单元其中一条边的索引
        cstart = mesh.ds.cellstart
        bnode2subdomain = np.zeros(NNB+NNC, dtype=np.int)
        bnode2subdomain[self.hedge2bnode] = halfedge[:, 1]
        if NNC >0:
            bnode2subdomain[-NNC:] = cstart # 将cnode对应的子区域记为内部区域
        self.bnode2subdomain = bnode2subdomain

        idx0 = halfedge[halfedge[:, 3], 0]
        idx1 = halfedge[:, 0]
        v = node[idx1] - node[idx0]
        h = np.sqrt(np.sum(v**2, axis=-1))

        if NNC > 0:
            bd = np.r_[self.bnode, self.cnode]
        else:
            bd = self.bnode
        tree = KDTree(bd)
        #c = 6*np.sqrt(3*(h[0]/2)*(h(0)/2)**3)
        c = 6*np.sqrt(3*(h[0]/2)*(h[0]/4)**3/2)
        self.inode = {} # 用一个字典来存储每个子区域的内部点
        for index in filter(lambda x: x > 0, self.mesh.ds.subdomain):
            p = bd[bnode2subdomain == index]
            xmin = min(p[:, 0])
            xmax = max(p[:, 0])
            ymin = min(p[:, 1])
            ymax = max(p[:, 1])
            area = self.mesh.cell_area(index)[index-1]
            N = int(area/c)
            N0 = p.shape[0]
            while N-N0 <= 0:
                c = 0.9*c
                N = int(area/c)
            start = 0
            newNode = np.zeros((N - N0, 2), dtype=node.dtype)
            NN = newNode.shape[0]
            i = 0
            while True:
                pp = np.random.rand(NN-start, 2)
                pp *= np.array([xmax-xmin,ymax-ymin])
                pp += np.array([xmin,ymin])
                d, idx = tree.query(pp)
                flag0 = d > (0.7*h[0])
                flag1 = (bnode2subdomain[idx] == cstart + index -1)
                pp = pp[flag0 & flag1]# 筛选出符合要求的点
                end = start + pp.shape[0]
                newNode[start:end] = pp
                if end == NN:
                    break
                else:
                    start = end
            self.inode[index] = newNode

    def voronoi(self):
        """
        对已经放置的种子点生成voronoi网格
        """
        bnode = self.bnode
        cnode = self.cnode
        inode = self.inode
       
        NNB = len(bnode) 
        NNC = len(cnode)
        NN = NNB + NNC
        for index, point in inode.items(): # inode is a dict
            NN += len(point)
        self.NN = NN
        points = np.zeros((NN, 2), dtype=bnode.dtype)
        points[:NNB] = bnode
        points[NNB:NNB+NNC] = cnode
        start = NNB + NNC
        for index, point in inode.items():
            N = len(point)
            points[start:start+N] = point
            start += N

        # construct voronoi diagram
        vor = Voronoi(points)
        start = NNB+NNC
        self.start = start
        return vor

    def lloyd_opt(self, vor):
        """
        Parameters
        ----------
        vor : 利用voronoi函数生成的vorornoi网格
        ----------
        该函数对voronoi网格进行一次lloyd优化
        """
        vertices = vor.vertices
        start = self.start # 边界处重构边界的种子点数量
        NN = self.NN
        rp = vor.ridge_points
        rv = np.array(vor.ridge_vertices)
        isKeeped = (rp[:, 0] >= start) | (rp[:, 1] >= start) 

        rp = rp[isKeeped]
        rv = rv[isKeeped]

        npoints = np.zeros((NN, 2), dtype=self.bnode.dtype)
        valence = np.zeros(NN, dtype=np.int)

        center = (vertices[rv[:, 0]] + vertices[rv[:, 1]])/2
        area = np.zeros(NN,dtype=np.float)
        np.add.at(npoints, (rp[:, 0], np.s_[:]), center)
        np.add.at(npoints, (rp[:, 1], np.s_[:]), center)
        np.add.at(valence, rp[:, 0], 1)
        np.add.at(valence, rp[:, 1], 1)
        npoints[start:] /= valence[start:, None]
        
        vor.points[start:, :] = npoints[start:, :]
        vor = Voronoi(vor.points)
        return vor
    
    def to_polygonMesh(self, vor):
        """
        Parameters
        ----------
        vor : 利用voronoi函数生成的voronoi网格
        ----------
        将voronoi网格转换为poolygonMesh数据结构
        """
        bnode2subdomain = self.bnode2subdomain
        start = self.start
        cstart = self.mesh.ds.cellstart
        points = vor.points
        pnode = vor.vertices
        point_region = vor.point_region
        ridge_vertices = vor.ridge_vertices

        bnode_region = point_region[:start]
        bnode_region = bnode_region[bnode2subdomain>=cstart]
        point_region = np.hstack((bnode_region, point_region[start:]))
        regions = [vor.regions[i] for i in point_region]
        pcellLocation =np.array([0] + [len(x) for x in regions],
                dtype=np.int_)
        np.cumsum(pcellLocation, out=pcellLocation)
        pcell = np.array(sum(regions,[]), dtype=np.int_)
        return PolygonMesh(pnode, pcell, pcellLocation)

    def energy(self,vor):
        """
        该函数计算voronoi网格的能量函数, 衡量网格质量
        """
        points = vor.points
        vertices = vor.vertices
        halfedge = self.mesh.entity('halfedge')

        NP = points.shape[0]
        area = np.zeros(NP,dtype=np.float)
        rp =vor.ridge_points
        rv = np.array(vor.ridge_vertices)
        isKeeped = (rv[:, 0]>=0)
        rp = rp[isKeeped]
        rv = rv[sKeeped]
        N = rp.shape[0]
        NN = 2*N
        p0 = np.zeros((NN,2),dtype = np.float)
        p1 = np.zeros((NN,2),dtype = np.float)
        p2 = np.zeros((NN,2),dtype = np.float)
        p0[:N] = points[rp[:,0]]
        p0[N:] = points[rp[:,1]]
        p1[:N] = vertices[rv[:,0]]
        p1[N:] = vertices[rv[:,1]]
        p2[:N] = vertices[rv[:,1]]
        p2[N:] = vertices[rv[:,0]]
        l0 = np.sqrt(np.sum((p1-p2)**2,axis=1))
        l1 = np.sqrt(np.sum((p0-p1)**2,axis=1))
        l2 = np.sqrt(np.sum((p0-p2)**2,axis=1))
        p = (l0+l1+l2)/2
        tri = np.sqrt(p*(p-l0)*(p-l1)*(p-l2))
        np.add.at(area,rp[:,0],tri[:N])
        np.add.at(area,rp[:,1],tri[N:])

        npoints = np.zeros((NP, 2), dtype=self.bnode.dtype)
        valence = np.zeros(NP, dtype=np.int)

        center = (vertices[rv[:, 0]] + vertices[rv[:, 1]])/2
        np.add.at(npoints, (rp[:, 0], np.s_[:]), center)
        np.add.at(npoints, (rp[:, 1], np.s_[:]), center)
        np.add.at(valence, rp[:, 0], 1)
        np.add.at(valence, rp[:, 1], 1)

        bp = self.hedge2bnode[halfedge[:,1]<=0]#无界区域和洞的种子点编号
        ap = np.ones(NP,dtype = np.bool)
        ap[bp] = False
        """
        pointflag = (valence ==0)
        valence[pointflag] +=1
        """
        npoints[:] /= valence[:,None]
        """
        npoints[~pointflag] = points[~pointflag]
        """
        energy = np.sum(np.sum((npoints[ap]-points[ap])**2,axis=1)*area[ap])
        aream = area[ap]
        ef = 2*(points[ap]-npoints[ap])*aream[:,None]
        return energy, ef

    def cell_area(self,index = np.s_[:]):
        return self.mesh.cell_area(index)

    def print(self):
        pass

        


        
