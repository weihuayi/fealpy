import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, tril, triu
from ..quadrature import TriangleQuadrature, QuadrangleQuadrature, GaussLegendreQuadrature 
from .Mesh2d import Mesh2d
from .adaptive_tools import mark
from .mesh_tools import show_halfedge_mesh
from ..common.Tools import hash2map
from ..common import DynamicArray


class HalfEdgeMesh2d(Mesh2d):
    def __init__(self, node, halfedge, subdomain, NV=None, nodedof=None):
        """
        这是一个用半边数据结构存储网格拓扑关系的类。半边数据结构表示的网格更适和
        网格的自适应算法的实现。

        Parameters
        ----------
        node : (NN, GD)
        halfedge : (2*NE, 4), 
            halfedge[i, 0]: the index of the vertex the i-th halfedge point to
            halfedge[i, 1]: the index of the cell the i-th halfedge blong to
            halfedge[i, 2]: the index of the next halfedge of i-th haledge 
            halfedge[i, 3]: the index of the prev halfedge of i-th haledge 
            halfedge[i, 4]: the index of the opposit halfedge of the i-th halfedge
        subdomain : (NC, ) the sub domain flag of each cell blong to
            单元所处的子区域的标记编号
             0: 表示外部无界区域
            -n: n >= 1, 表示编号为 -n 洞
             n: n >= 1, 表示编号为  n 的内部子区域

        Notes
        -----
        这个类的核心数组都是动态数组， 可以根据网格实体数目的变化动态增加长度，
        理论上可有效减少内存开辟的次数。

        Reference
        ---------
        [1] https://github.com/maciejkula/dynarray/blob/master/dynarray/dynamic_array.py

        """

        self.itype = halfedge.dtype
        self.ftype = node.dtype

        self.node = DynamicArray(node, dtype = node.dtype)
        self.ds = HalfEdgeMesh2dDataStructure(halfedge, 
                subdomain, NN = node.shape[0], NV=NV)
        self.meshtype = 'halfedge2d'

        self.halfedgedata = {}
        self.celldata = {}
        self.nodedata = {}
        self.edgedata = {}
        self.facedata = self.edgedata
        self.meshdata = {}

        # 网格节点的自由度标记数组
        # 0: 固定点
        # 1: 边界上的点
        # 2: 区域内部的点
        self.nodedata['dof'] = nodedof

        self.init_level_info()

    @classmethod
    def from_mesh(cls, mesh):
        mtype = mesh.meshtype
        if mtype != 'halfedge':
            NE = mesh.number_of_edges()
            NC = mesh.number_of_cells()

            node = mesh.entity('node')
            edge = mesh.entity('edge')
            edge2cell = mesh.ds.edge_to_cell()
            isInEdge = edge2cell[:, 0] != edge2cell[:, 1]
            

            halfedge = np.zeros((2*NE, 5), dtype=mesh.itype)
            halfedge[:, 0] = edge.flat

            halfedge[0::2, 1][isInEdge] = edge2cell[isInEdge, 1] + 1
            halfedge[1::2, 1] = edge2cell[:, 0] + 1

            halfedge[0::2, 4] = range(1, 2*NE, 2)
            halfedge[1::2, 4] = range(0, 2*NE, 2)

            NHE = len(halfedge)
            edge = np.zeros((2*NHE, 2), dtype=halfedge.dtype)
            edge[:NHE] = halfedge[:, 0:2]
            edge[NHE:, 0] = halfedge[halfedge[:, 4], 0]
            edge[NHE:, 1] = halfedge[:, 1]
            idx = np.lexsort((edge[:, 0], edge[:, 1])).reshape(-1, 2)
            idx[:, 1] -= NHE
            halfedge[idx[:, 0], 2] = idx[:, 1]
            halfedge[halfedge[:, 2], 3] = range(NHE)

            subdomain = np.ones(NC+1, dtype=halfedge.dtype)
            subdomain[0] = 0
            return cls(node, halfedge, subdomain)
        else:
            newMesh =  cls(mesh.node, mesh.subdomain, mesh.ds.halfedge.copy())
            newMesh.celldata['level'][:] = mesh.celldata['level']
            newMesh.nodedata['level'][:] = mesh.nodedata['level']
            newMesh.halfedge['level'][:] = mesh.halfedgedata['level']
            return newMesh

    @classmethod
    def from_edges(cls, node, edge, edge2subdomain, nodedof=None):
        """
        Parameters
        ----------
        node : (NN, GD)
        edge : (NF, 2)
        edge2subdomain : (NF, 2)
                 0: 表示外部无界区域
                -n: n >= 1, 表示编号为 -n 洞
                 n: n >= 1, 表示编号为  n 的内部子区域
        Examples
        -------
        >> 
        """

        NN = len(node)
        NE = len(edge)

        halfedge = np.zeros((2*NE, 5), dtype=edge.dtype)
        halfedge[:, 0] = edge.flat

        halfedge[0::2, 1] = edge2subdomain[:, 1]
        halfedge[1::2, 1] = edge2subdomain[:, 0] 
        cell2subdomain, _, j = np.unique(halfedge[:, 1], return_index=True, return_inverse=True)
        halfedge[:, 1] = j
        
        halfedge[0::2, 4] = range(1, 2*NE, 2)
        halfedge[1::2, 4] = range(0, 2*NE, 2)

        NHE = len(halfedge)
        facets = np.zeros((2*NHE, 2), dtype=edge.dtype)
        facets[:NHE] = halfedge[:, 0:2]
        facets[NHE:, 0] = halfedge[halfedge[:, 4], 0]
        facets[NHE:, 1] = halfedge[:, 1]
        idx = np.lexsort((facets[:, 0], facets[:, 1])).reshape(-1, 2)
        idx[:, 1] -= NHE
        halfedge[idx[:, 0], 2] = idx[:, 1]
        halfedge[halfedge[:, 2], 3] = range(NHE)

        return cls(node, halfedge, cell2subdomain) 

    @classmethod
    def from_poly(self):
        """

        Reference
        ---------
        [.poly files] https://www.cs.cmu.edu/~quake/triangle.poly.html
        """
        pass

    def init_level_info(self):
        NN = self.number_of_nodes()
        NE = self.number_of_edges()

        NC = self.number_of_all_cells() # 实际单元个数
        self.halfedgedata['level'] = DynamicArray((2*NE, ), val=0, dtype=np.uint8)
        self.celldata['level'] = DynamicArray((NC, ), val=0, dtype=np.uint8) 
        self.nodedata['level'] = DynamicArray((NN, ), val=0, dtype=np.uint8)

    def number_of_all_cells(self):
        return self.ds.number_of_all_cells()

    def set_data(self, name, val, etype):
        if etype in {'cell', 2}:
            NC = self.number_of_all_cells() 
            shape = (NC, ) + val.shape[1:]
            self.celldata[name] = np.zeros(shape, dtype=val.dtype) 
            self.celldata[name][self.ds.cellstart:] = val
        elif etype in {'face', 'edge', 1}:
            self.edgedata[name] = val
        elif etype in {'node', 0}:
            self.nodedata[name] = val
        elif etype == 'mesh':
            self.meshdata[name] = val
        elif etype == 'halfedge':
            self.halfedgedata[name] = val
        else:
            raise ValueError("`etype` is wrong!")

    def get_data(self, etype, name):
        if etype in {'cell', 2}:
            return self.celldata[name][self.ds.cellstart:]
        elif etype in {'face', 'edge', 1}:
            return self.edgedata[name]
        elif etype in {'node', 0}:
            return self.nodedata[name]
        elif etype == 'mesh':
            return self.meshdata[name]
        elif etype == 'halfedge':
            return self.halfedgedata[name]
        else:
            raise ValueError("`etype` is wrong!")

    def integrator(self, k, etype='cell'):
        if etype in {'cell', 'tri',  2}:
            return TriangleQuadrature(k)
        elif etype in {'quad'}:
            return QuadrangleQuadrature(k)
        elif etype in {'edge', 'face', 1}:
            return GaussLegendreQuadrature(k)


    def entity(self, etype=2):
        if etype in {'cell', 2}:
            return self.ds.cell_to_node()
        elif etype in {'edge', 'face', 1}:
            return self.ds.edge_to_node()
        elif etype in {'halfedge'}:
            return self.ds.halfedge # DynamicArray
        elif etype in {'node', 0}:
            return self.node # DynamicArrray
        else:
            raise ValueError("`entitytype` is wrong!")

    def entity_barycenter(self, etype='cell', index=None):
        node = self.entity('node')
        GD = self.geo_dimension()
        if etype in {'cell', 2}:
            # 这里是单元顶点坐标的平均位置
            cell2node = self.ds.cell_to_node(return_sparse=True)
            if self.ds.NV is None:
                NV = self.ds.number_of_vertices_of_cells()
                bc = cell2node@node/NV[:, None]
            else:
                bc = cell2node@node/NV
        elif etype in {'edge', 'face', 1}:
            edge = self.ds.edge_to_node()
            bc = np.sum(node[edge, :], axis=1).reshape(-1, GD)/edge.shape[1]
        elif etype in {'node', 0}:
            bc = node
        return bc

    def node_normal(self):
        node = self.node
        cell = self.entity('cell') 
        if isinstance(cell, tuple):
            cell, cellLocation = cell
            idx1 = np.zeros(cell.shape[0], dtype=np.int)
            idx2 = np.zeros(cell.shape[0], dtype=np.int)

            idx1[0:-1] = cell[1:]
            idx1[cellLocation[1:]-1] = cell[cellLocation[:-1]]
            idx2[1:] = cell[0:-1]
            idx2[cellLocation[:-1]] = cell[cellLocation[1:]-1]
            w = np.array([(0,-1),(1,0)])
            d = node[idx1] - node[idx2]
            return 0.5*d@w
        else:
            assert self.ds.NV == 3 or self.ds.NV == 4
            # TODO: for tri and quad case

    def cell_area(self, index=None):
        NC = self.number_of_cells()
        node = self.entity('node')

        halfedge = self.ds.halfedge # DynamicArray
        hflag = self.ds.subdomain[halfedge[:, 1]] > 0
        cstart = self.ds.cstart

        e0 = halfedge[halfedge[hflag, 3], 0]
        e1 = halfedge[hflag, 0]

        w = np.array([[0, -1], [1, 0]], dtype=np.int)
        v = (node[e1] - node[e0])@w
        val = np.sum(v*node[e0], axis=1)

        a = np.zeros(NC, dtype=self.ftype)
        np.add.at(a, halfedge[hflag, 1] - cstart, val)
        a /=2
        return a

    def cell_barycenter(self, return_all=False):
        """
        这里是单元的物理重心。

        Parameters
        ----------

        Notes
        -----

        """
        GD = self.geo_dimension()
        node = self.entity('node') # DynamicArray
        halfedge = self.entity('halfedge') # DynamicArray
        if return_all:
            NC = self.number_of_all_cells()
            e0 = halfedge[halfedge[:, 3], 0]
            e1 = halfedge[:, 0]
            w = np.array([[0, -1], [1, 0]], dtype=np.int)
            v= (node[e1] - node[e0])@w
            val = np.sum(v*node[e0], axis=1)
            ec = val.reshape(-1, 1)*(node[e1]+node[e0])/2

            a = np.zeros(NC, dtype=self.ftype)
            c = np.zeros((NC, GD), dtype=self.ftype)
            np.add.at(a, halfedge[:, 1], val)
            np.add.at(c, (halfedge[:, 1], np.s_[:]), ec)
            a /=2
            c /=3*a.reshape(-1, 1)
            return c
        else:
            NC = self.number_of_cells()
            hflag = self.ds.subdomain[halfedge[:, 1]] > 0
            cstart = self.ds.cellstart
            e0 = halfedge[halfedge[hflag, 3], 0]
            e1 = halfedge[hflag, 0]
            w = np.array([[0, -1], [1, 0]], dtype=np.int)
            v= (node[e1] - node[e0])@w
            val = np.sum(v*node[e0], axis=1)
            ec = val.reshape(-1, 1)*(node[e1]+node[e0])/2
            a = np.zeros(NC, dtype=self.ftype)
            c = np.zeros((NC, GD), dtype=self.ftype)
            np.add.at(a, halfedge[hflag, 1] - cstart, val)
            np.add.at(c, (halfedge[hflag, 1] - cstart, np.s_[:]), ec)
            a /=2
            c /=3*a.reshape(-1, 1)
            return c

    def bc_to_point(self, bc, etype='cell', index=None):
        """

        Parameters
        ----------
        bc : (3, ) or (NQ, 3)
        etype : 'cell' or 'edge'
        """
        assert self.ds.NV == 3
        node = self.entity('node')
        entity = self.entity(etype) # default  cell
        index = index if index is not None else np.s_[:]
        p = np.einsum('...j, ijk->...ik', bc, node[entity[index]])
        return p

    def edge_bc_to_point(self, bcs, index=None):
        """
        """
        node = self.entity('node')
        edge = self.entity('edge')
        index = index if index is not None else np.s_[:]
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge[index]])
        return ps

    def mark_halfedge(self, isMarkedCell, method='poly'):
        clevel = self.celldata['level'] # 注意这里是所有的单元的层信息
        nlevel = self.nodedata['level']
        hlevel = self.halfedgedata['level']
        halfedge = self.entity('halfedge')
        if method == 'poly':
            # 当前半边的层标记小于等于所属单元的层标记
            flag0 = (hlevel - clevel[halfedge[:, 1]]) <= 0 
            # 前一半边的层标记小于等于所属单元的层标记 
            pre = halfedge[:, 3]
            flag1 = (hlevel[pre] - clevel[halfedge[:, 1]]) <= 0
            # 标记加密的半边
            isMarkedHEdge = isMarkedCell[halfedge[:, 1]] & flag0 & flag1 
            # 标记加密的半边的相对半边也需要标记 
            flag = ~isMarkedHEdge & isMarkedHEdge[halfedge[:, 4]]
            isMarkedHEdge[flag] = True
        elif method == 'quad':
            pass
        elif method == 'rg':
            pass
        elif method == 'rgb':
            pass
        return isMarkedHEdge

    def refine_halfedge(self, isMarkedHEdge):

        NN = self.number_of_nodes() 
        NE = self.number_of_edges() 

        clevel = self.celldata['level']
        hlevel = self.halfedgedata['level'] 

        halfedge = self.entity('halfedge')
        node = self.entity('node')
        hedge = self.ds.hedge
        hcell = self.ds.hcell
        subdomain = self.ds.subdomain

        isMainHEdge = self.ds.main_halfedge_flag()


        # 即是主半边, 也是标记加密的半边
        flag0 = isMarkedHEdge & isMainHEdge
        idx = halfedge[flag0, 4]

        NE1 = flag0.sum()
        newNode = node.increase_size(NE1)
        newNode[:] = (node[halfedge[flag0, 0]] + node[halfedge[idx, 0]])/2

        #细分边
        newHalfedge = halfedge.increase_size(2*NE1)
        newHlevel = hlevel.increase_size(2*NE1)

        flag1 = np.cumsum(isMarkedHEdge)[flag0]-1 # 标记加密边中的主半边, flag1是加密的主半边 flag0 在所有需要加密的边中的编号
        flag2 = np.cumsum(isMarkedHEdge)[idx]-1
        halfedge1[flag1, 0] = range(NN, NN+NE1) # 新的节点编号
        halfedge1[flag2, 0] = range(NN, NN+NE1)
        nlevel1 = np.zeros(NE1, dtype=self.itype)
        nlevel1 = np.maximum(nlevel[halfedge[flag0, 0]], 
                nlevel[halfedge[halfedge[flag0, 3], 0]]) + 1

        halfedge1[:, 1] = halfedge[isMarkedHEdge, 1]
        halfedge1[:, 3] = halfedge[isMarkedHEdge, 3] # 前一个 
        halfedge1[:, 4] = halfedge[isMarkedHEdge, 4] # 对偶边

        #hedge.extend(flag1+NE*2)
        halfedge[isMarkedHEdge, 3] = range(2*NE, 2*NE + 2*NE1)
        idx = halfedge[isMarkedHEdge, 4] # 原始对偶边
        halfedge[isMarkedHEdge, 4] = halfedge[idx, 3]  # 原始对偶边的前一条边是新的对偶边

        halfedge.extend(halfedge1)
        halfedge[halfedge[:, 3], 2] = range(2*NE+2*NE1)
        nlevel = np.r_[nlevel, nlevel1]



    def add_halfedge_plot(self, axes,
        index=None, showindex=False,
        nodecolor='r', edgecolor=['r', 'k'], markersize=20,
        fontsize=20, fontcolor='k', multiindex=None, linewidth=0.5):

        show_halfedge_mesh(axes, self,
                index=index, showindex=showindex,
                nodecolor=nodecolor, edgecolor=edgecolor, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor, 
                multiindex=multiindex, linewidth=linewidth)

    def print(self):
        print("hcell:\n")
        for i, val in enumerate(self.ds.hcell):
            print(i, ':', val)

        print("hedge:")
        for i, val in enumerate(self.ds.hedge):
            print(i, ":", val)

        print("halfedge:")
        for i, val in enumerate(self.ds.halfedge):
            print(i, ":", val)

class HalfEdgeMesh2dDataStructure():
    def __init__(self, halfedge, subdomain, NN=None, NV=None):
        self.reinit(halfedge, subdomain, NN=NN, NV=NV)

    def reinit(self, halfedge, subdomain, NN=None, NV=None):
        """

        Note
        ----
        self.halfedge, self.subdomain, self.hcell, self.hedge are DynamicArray
        """

        self.itype = halfedge.dtype

        self.halfedge = DynamicArray(halfedge, dtype=self.itype)
        self.subdomain = DynamicArray(subdomain, dtype=self.itype)

        self.NN = NN
        self.NE = len(halfedge)//2
        self.NF = self.NE

        # 区域内部的单元标记, 这里默认排前面的都是洞, 或者外部无界区域.
        idx, = np.nonzero(subdomain == 0)
        if len(idx) == 0:
            self.cellstart = 0 
        elif len(idx) == 1:
            self.cellstart = idx[0] + 1
        else:
            raise ValueError("The number of unbounded doamin is bigger than 1!")

        self.NC = len(subdomain) - self.cellstart # 区域内单元的个数

        NC = len(subdomain) # 实际单元个数, 包括外部无界区域和洞

        self.hcell = DynamicArray( (NC, ), dtype=self.itype) # hcell[i] is the index of one face of i-th cell

        self.hcell[halfedge[:, 1]] = range(2*self.NE) # 的编号
        flag = halfedge[:, 4] - np.arange(2*self.NE) > 0
        self.hedge = DynamicArray(np.arange(self.NE*2)[flag])
        flag = subdomain[halfedge[self.hedge, 1]] < 1
        self.hedge[flag] = halfedge[self.hedge[flag], 4]

        self.NV = NV

    def number_of_all_cells(self):
        return len(self.subdomain)

    def number_of_vertices_of_all_cells(self):
        NC = self.number_of_all_cells() 
        halfedge = self.halfedge
        NV = np.zeros(NC, dtype=self.itype)
        np.add.at(NV, halfedge[:, 1], 1)
        return NV

    def number_of_vertices_of_cells(self):
        if self.NV in {3, 4}:
            return self.NV
        else:
            NC = self.NC 
            halfedge = self.halfedge
            subdomain = self.subdomain
            NV = np.zeros(NC, dtype=self.itype)
            flag =  subdomain[halfedge[:, 1]] > 0
            np.add.at(NV, halfedge[flag, 1]-self.cellstart, 1)
            return NV

    def number_of_nodes_of_cells(self):
        return self.number_of_vertices_of_cells()

    def number_of_edges_of_cells(self):
        return self.number_of_vertices_of_cells()

    def number_of_faces_of_cells(self):
        return self.number_of_vertices_of_cells()

    def cell_to_node(self, return_sparse=False):
        NN = self.NN
        NC = self.NC
        halfedge = self.halfedge
        subdomain = self.subdomain
        cstart = self.cellstart
        hflag = subdomain[halfedge[:, 1]] > 0

        if return_sparse:
            val = np.ones(hflag.sum(), dtype=np.bool)
            I = halfedge[hflag, 1] - cstart
            J = halfedge[hflag, 0]
            cell2node = csr_matrix((val, (I, J)), shape=(NC, NN), dtype=np.bool)
            return cell2node
        elif self.NV is None: # polygon mesh
            NV = self.number_of_vertices_of_cells()
            cellLocation = np.zeros(NC+1, dtype=self.itype)
            cellLocation[1:] = np.cumsum(NV)
            cell2node = np.zeros(cellLocation[-1], dtype=self.itype)
            current = self.hcell[cstart:]
            idx = cellLocation[:-1].copy()
            cell2node[idx] = halfedge[halfedge[current, 3], 0]
            NV0 = np.ones(NC, dtype=self.itype)
            isNotOK = NV0 < NV
            while np.any(isNotOK):
               cell2node[idx[isNotOK]] = halfedge[current[isNotOK], 0]
               idx[isNotOK] += 1
               NV0[isNotOK] += 1
               current[isNotOK] = halfedge[current[isNotOK], 2]
               isNotOK = (NV0 < NV)
            return cell2node, cellLocation
        elif self.NV == 3: # tri mesh
            cell2node = np.zeros(NC, 3)
            current = halfedge[self.hcell[cstart:], 2]
            cell2node[:, 0] = halfedge[current, 0]
            current = halfedge[current, 2]
            cell2node[:, 1] = halfedge[current, 0]
            current = halfedge[current, 2]
            cell2node[:, 2] = halfedge[current, 0]
            return cell2node
        elif self.NV == 4: # quad mesh
            cell2node = np.zeros(NC, 3)
            current = halfedge[self.hcell[cstart:], 3]
            cell2node[:, 0] = halfedge[current, 0]
            current = halfedge[current, 2]
            cell2node[:, 1] = halfedge[current, 0]
            current = halfedge[current, 2]
            cell2node[:, 2] = halfedge[current, 0]
            current = halfedge[current, 2]
            cell2node[:, 3] = halfedge[current, 0]
            return cell2node
        else:
            raise ValueError('The property NV should be None, 3 or 4! But the NV is {}'.format(self.NV))

    def cell_to_edge(self, return_sparse=False):
        NE = self.NE
        NC = self.NC

        halfedge = self.halfedge
        cstart = self.cellstart
        hflag = subdomain[halfedge[:, 1]] > 0
        hedge = self.hedge

        J = np.zeros(2*NE, dtype=self.itype)
        J[hedge] = range(NE)
        J[halfedge[hedge, 4]] = range(NE)
        if return_sparse:
            val = np.ones(2*NE, dtype=np.bool)
            I = halfedge[hflag, 1] - cstart
            cell2edge = csr_matrix((val[hflag], (I,
                J[hflag])), shape=(NC, NE), dtype=np.bool)
            return cell2edge
        elif NV is None:
            NV = self.number_of_vertices_of_cells()
            cellLocation = np.zeros(NC+1, dtype=self.itype)
            cellLocation[1:] = np.cumsum(NV)

            cell2edge = np.zeros(cellLocation[-1], dtype=self.itype)
            current = self.hcell[cstart:]
            idx = cellLocation[:-1]
            cell2edge[idx] = J[current]
            NV0 = np.ones(NC, dtype=self.itype)
            isNotOK = NV0 < self.NV
            while isNotOK.sum() > 0:
                current[isNotOK] = halfedge[current[isNotOK], 2]
                idx[isNotOK] += 1
                NV0[isNotOK] += 1
                cell2edge[idx[isNotOK]] = J[current[isNotOK]]
                isNotOK = (NV0 < self.NV)
            return cell2edge, cellLocation
        elif self.NV == 3: # tri mesh
            cell2edge = np.zeros(NC, 3)
            current = self.hcell[cstart:]
            cell2edge[:, 2] = J[current]
            cell2edge[:, 0] = J[halfedge[current, 2]]
            cell2edge[:, 1] = J[halfedge[current, 3]]
            return cell2edge
        elif self.NV == 4: # quad mesh
            cell2edge = np.zeros(NC, 4)
            current = self.hcell[cstart:]
            cell2edge[:, 3] = J[current]
            current = halfedge[current, 2]
            cell2edge[:, 0] = J[current] 
            current = halfedge[current, 2]
            cell2edge[:, 1] = J[current]
            current = halfedge[current, 2]
            cell2edge[:, 2] = J[current]
            return cell2edge
        else:
            raise ValueError('The property NV should be None, 3 or 4! But the NV is {}'.format(self.NV))

    def cell_to_face(self, return_sparse=True):
        return self.cell_to_edge(return_sparse=return_sparse)

    def cell_to_cell(self, return_sparse=True):
        NC = self.NC
        halfedge = self.halfedge
        cstart = self.cellstart
        hflag = subdomain[halfedge[:, 1]] > 0
        hedge = self.hedge

        if return_sparse:
            flag = hflag & hflag[halfedge[:, 4]]
            val = np.ones(flag.sum(), dtype=np.bool_)
            I = halfedge[flag, 1]
            J = halfedge[halfedge[flag, 4], 1]
            cell2cell = coo_matrix((val, (I, J)), shape=(NC, NC), dtype=np.bool)
            cell2cell+= coo_matrix((val, (J, I)), shape=(NC, NC), dtype=np.bool)
            return cell2cell.tocsr()
        elif self.NV is None:
            NV = self.number_of_vertices_of_cells()
            cellLocation = np.zeros(NC+1, dtype=self.itype)
            cellLocation[1:] = np.cumsum(NV)
            cell2cell = np.zeros(cellLocation[-1], dtype=self.itype)
            current = self.hcell[cstart:]
            idx = cellLocation[:-1]
            cell2cell[idx] = halfedge[halfedge[current, 4], 1]-cstart
            NV0 = np.ones(NC, dtype=self.itype)
            isNotOK = NV0 < NV
            while isNotOK.sum() > 0:
                current[isNotOK] = halfedge[current[isNotOK], 2]
                idx[isNotOK] += 1
                NV0[isNotOK] += 1
                cell2cell[idx[isNotOK]] = halfedge[halfedge[current[isNotOK], 4], 1]-cstart
                isNotOK = (NV0 < NV)
            idx = np.repeat(range(NC), NV)
            flag = (cell2cell < 0)
            cell2cell[flag] = idx[flag]
            return cell2cell, cellLocation
        elif self.NV == 3: # tri mesh
            cell2cell = np.zeros((NC, 3), dtype=self.itype)
            current = self.hcell[cstart:]
            cell2cell[:, 0] = halfedge[halfedge[current, 4], 1] - cstart
            cell2cell[:, 1] = halfedge[halfedge[halfedge[current, 2], 4], 1] - cstart
            cell2cell[:, 2] = halfedge[halfedge[halfedge[current, 3], 4], 1] - cstart
            idx = np.repeat(range(NC), 3).reshape(NC, 3)
            flag = (cell2cell < 0)
            cell2cell[flag] = idx[flag]
            return cell2cella
        elif self.NV == 4: # quad mesh
            cell2cell = np.zeros(NC, 4)
            current = self.hcell[cstart:]
            cell2cell[:, 3] = halfedge[halfedge[current, 4], 1] - cstart 
            current = halfedge[current, 2]
            cell2cell[:, 0] = halfedge[halfedge[current, 4], 1] - cstart
            current = halfedge[current, 2]
            cell2cell[:, 1] = halfedge[halfedge[current, 4], 1] - cstart
            current = halfedge[current, 2]
            cell2cell[:, 2] = halfedge[halfedge[current, 4], 1] - cstart
            idx = np.repeat(range(NC), 4).reshape(NC, 4)
            flag = (cell2cell < 0)
            cell2cell[flag] = idx[flag]
            return cell2cell
        else:
            raise ValueError('The property NV should be None, 3 or 4! But the NV is {}'.format(self.NV))

    def edge_to_node(self, return_sparse=False):
        NN = self.NN
        NE = self.NE
        halfedge = self.halfedge
        hedge = self.hedge
        if return_sparse == False:
            edge = np.zeros((NE, 2), dtype=self.itype)
            edge[:, 0] = halfedge[halfedge[hedge, 4], 0]
            edge[:, 1] = halfedge[hedge, 0]
            return edge
        else:
            val = np.ones(NE, dtype=np.bool_)
            edge2node = coo_matrix((val, (range(NE), halfedge[hedge, 0])),
                    shape=(NE, NN), dtype=np.bool_)
            edge2node+= coo_matrix(
                    (val, (range(NE), halfedge[halfedge[hedge, 4], 0])),
                    shape=(NE, NN), dtype=np.bool_)
            return edge2node.tocsr()

    def edge_to_edge(self):
        edge2node = self.edge_to_node()
        return edge2node*edge2node.tranpose()

    def edge_to_cell(self):
        NE = self.NE
        NC = self.NC

        halfedge = self.halfedge
        cstart = self.cellstart
        subdomain = self.subdomain
        hflag = subdomain[halfedge[:, 1]] > 0
        hedge = self.hedge

        J = np.zeros(2*NE, dtype=self.itype)
        J[hedge] = range(NE)
        J[halfedge[hedge, 4]] = range(NE)

        edge2cell = np.full((NE, 4), -1, dtype=self.itype)
        edge2cell[J[hedge], 0] = halfedge[hedge, 1] - cstart
        edge2cell[J[halfedge[hedge, 4]], 1] = halfedge[halfedge[hedge, 4], 1] - cstart

        isMainHEdge = np.zeros(2*NE, dtype=np.bool_)
        isMainHEdge[hedge] = True

        if self.NV is None:
            current = self.hcell[cstart:]
            end = current.copy()
            lidx = np.zeros_like(current)
            isNotOK = np.ones_like(current, dtype=np.bool_)
            while np.any(isNotOK):
                idx = J[current[isNotOK]]
                flag = isMainHEdge[current[isNotOK]]
                edge2cell[idx[flag], 2] = lidx[isNotOK][flag]
                edge2cell[idx[~flag], 3] = lidx[isNotOK][~flag]
                current[isNotOK] = halfedge[current[isNotOK], 2]
                lidx[isNotOK] += 1
                isNotOK = (current != end)
        elif self.NV == 3:
            current = self.hcell[cstart:]
            idx = J[current]
            flag = isMainHEdge[current] 
            edge2cell[idx[flag], 2] = 0
            edge2cell[idx[~flag], 3] = 0

            idx = J[halfedge[current, 2]]
            flag = isMainHEdge[halfedge[current, 2]] 
            edge2cell[idx[flag], 2] = 1
            edge2cell[idx[~flag], 3] = 1

            idx = J[halfedge[current, 3]]
            flag = isMainHEdge[halfedge[current, 3]] 
            edge2cell[idx[flag], 2] = 2
            edge2cell[idx[~flag], 3] = 2
        elif self.NV == 4:
            current = self.hcell[cstart]
            idx = J[current]
            flag = isMainHEdge[current] 
            edge2cell[idx[flag], 2] = 0
            edge2cell[idx[~flag], 3] = 0

            current = halfedge[current, 2]
            idx = J[current]
            flag = isMainHEdge[current] 
            edge2cell[idx[flag], 2] = 1
            edge2cell[idx[~flag], 3] = 1

            current = halfedge[current, 2]
            idx = J[current]
            flag = isMainHEdge[current] 
            edge2cell[idx[flag], 2] = 2
            edge2cell[idx[~flag], 3] = 2

            current = halfedge[current, 2]
            idx = J[current]
            flag = isMainHEdge[current] 
            edge2cell[idx[flag], 2] = 3
            edge2cell[idx[~flag], 3] = 3
        else:
            raise ValueError('The property NV should be None, 3 or 4! But the NV is {}'.format(self.NV))

        flag = edge2cell[:, 1] < 0 
        edge2cell[flag, 1] = edge2cell[flag, 0]
        edge2cell[flag, 3] = edge2cell[flag, 2]
        return edge2cell

    def node_to_node(self, return_sparse=True):
        NN = self.NN
        NE = self.NE
        halfedge = self.halfedge
        I = halfedge[:, 0] 
        J = halfedge[halfedge[:, 4], 0] 
        val = np.ones(2*NE, dtype=np.bool)
        node2node = csr_matrix((val, (I, J)), shape=(NN, NN), dtype=np.bool_)
        return node2node

    def node_to_edge(self, return_sparse=True):
        pass

    def node_to_cell(self, return_sparse=True):
        NN = self.NN
        NC = self.NC
        halfedge =  self.halfedge
        subdomain = self.subdomain
        cstart = self.cellstart
        hflag = subdomain[halfedge[:, 1]] > 0

        val = np.ones(hflag.sum(), dtype=np.bool_)
        I = halfedge[hflag, 0]
        J = halfedge[hflag, 1] - cstart
        node2cell = csr_matrix((val, (I.flat, J.flat)), shape=(NN, NC), dtype=np.bool_)
        return node2cell

    def boundary_node_flag(self):
        NN = self.NN
        halfedge =  self.halfedge # DynamicArray
        subdomain = self.subdomain # DynamicArray
        hflag = subdomain[halfedge[:, 1]] > 0
        isBdHEdge = hflag & (~hflag[halfedge[:, 4]])
        isBdNode = np.zeros(NN, dtype=np.bool)
        isBdNode[halfedge[isBdHEdge, 0]] = True 
        return isBdNode

    def boundary_edge_flag(self):
        NE = self.NE
        halfedge =  self.halfedge
        subdomain = self.subdomain
        hflag = subdomain[halfedge[:, 1]] > 0
        isBdHEdge = hflag & (~hflag[halfedge[:, 4]])
        J = np.zeros(2*NE, dtype=self.itype)
        J[hedge] = range(NE)
        J[halfedge[hedge, 4]] = range(NE)
        return J[isBdHEdge] 

    def boundary_edge(self):
        edge = self.edge_to_node()
        return edge[self.boundary_edge_index()]

    def boundary_cell_flag(self):
        """

        Parameters
        ----------

        Notes
        -----

        Reference
        ---------
        """
        NC = self.NC
        cstart = self.cellstart
        halfedge =  self.halfedge # DynamicArray
        subdomain = self.subdomain # DynamicArray
        hflag = subdomain[halfedge[:, 1]] > 0
        isBdHEdge = hflag & (~hflag[halfedge[:, 4]])

        isBdCell = np.zeros(NC, dtype=np.bool)
        idx = halfedge[isBdHEdge, 1] - cstart
        isBdCell[idx] = True
        return isBdCell

    def boundary_node_index(self):
        isBdNode = self.boundary_node_flag()
        idx, = np.nonzero(isBdNode)
        return idx

    def boundary_edge_index(self):
        isBdEdge = self.boundary_edge_flag()
        idx, = np.nonzero(isBdEdge)
        return idx

    def boundary_cell_index(self):
        isBdCell = self.boundary_cell_flag()
        idx, = np.nonzero(isBdCell)
        return idx

    def main_halfedge_flag(self):
        isMainHEdge = np.zeros(2*self.NE, dtype=np.bool)
        isMainHEdge[self.hedge] = True
        return isMainHEdge 
