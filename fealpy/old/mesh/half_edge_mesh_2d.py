"""


Notes
-----

Authors
-------
    Huayi Wei, Chunyu Chen, Xin Wang
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, tril, triu
from scipy.sparse.csgraph import minimum_spanning_tree

from ..quadrature import TriangleQuadrature, QuadrangleQuadrature, GaussLegendreQuadrature 
from .mesh_base import Mesh, Plotable
from .adaptive_tools import mark
from .mesh_tools import show_halfedge_mesh
from fealpy.common import DynamicArray
from .triangle_mesh import TriangleMesh


class HalfEdgeMesh2d(Mesh, Plotable):
    def __init__(self, node, halfedge, subdomain, NV=None, nodedof=None,
            initlevel=True):
        """

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
        这是一个用半边数据结构存储网格拓扑关系的类。半边数据结构表示的网格更适和
        网格的自适应算法的实现。

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
        self.hedgecolor = None

        self.newnode2edge = {}
        self.retainnode = {}
        self.deletnode2edge = {}

        # 网格节点的自由度标记数组
        # 0: 固定点
        # 1: 边界上的点
        # 2: 区域内部的点
        if nodedof is not None:
            self.nodedata['dof'] = nodedof

        if initlevel:
            self.init_level_info()

    @classmethod
    def from_mesh(cls, mesh, closed=False, NV=None):
        """

        Notes
        -----
        输入一个其它类型数据结构的网格，转化为半边数据结构。如果 closed 为真，则
        表明输入网格是一个封闭的曲面网格；为假，则为开的网格，可以存在洞，或者无
        界的外界区域
        """
        mtype = mesh.meshtype
        if mtype not in {'halfedge', 'halfedge2d'}:
            NE = mesh.number_of_edges()
            NC = mesh.number_of_cells()

            node = mesh.entity('node')
            edge = mesh.entity('edge')
            edge2cell = mesh.ds.edge_to_cell()
            isInEdge = edge2cell[:, 0] != edge2cell[:, 1]


            halfedge = np.zeros((2*NE, 5), dtype=mesh.itype)
            halfedge[:, 0] = edge.flat

            if closed:
                halfedge[0::2, 1][isInEdge] = edge2cell[isInEdge, 1]
                halfedge[1::2, 1] = edge2cell[:, 0]
            else:
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

            if closed:
                subdomain = np.ones(NC, dtype=halfedge.dtype)
            else:
                subdomain = np.ones(NC+1, dtype=halfedge.dtype)
                subdomain[0] = 0

            mesh =  cls(node, halfedge, subdomain, NV=NV)
            return mesh
        else:
            newMesh =  cls(mesh.node.copy(), mesh.ds.subdomain.copy(), mesh.ds.halfedge.copy())
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

        self.halfedgedata['level'] = DynamicArray((2*NE, ), val=0,dtype=np.int_)
        self.celldata['level'] = DynamicArray((NC, ), val=0, dtype=np.int_) 
        self.nodedata['level'] = DynamicArray((NN, ), val=0, dtype=np.int_)

        # 如果单元的角度大于 170 度， 设对应的半边层数为 1
        node = self.node
        halfedge = self.ds.halfedge
        v0 = node[halfedge[halfedge[:, 2], 0]] - node[halfedge[:, 0]]
        v1 = node[halfedge[halfedge[:, 4], 0]] - node[halfedge[:, 0]]

        angle = np.sum(v0*v1, axis=1)/np.sqrt(np.sum(v0**2, axis=1)*np.sum(v1**2, axis=1))
        self.halfedgedata['level'][
                (angle < -0.98) & (halfedge[:, 1] >= self.ds.cellstart)
                ] = 1 

    def convexity(self):#
        def angle(x, y):
            x = x/(np.linalg.norm(x, axis=1)).reshape(len(x), 1)
            y = y/np.linalg.norm(y, axis=1).reshape(len(y), 1)
            theta = np.sign(np.cross(x, y))*np.arccos((x*y).sum(axis=1))
            theta[theta<0]+=2*np.pi
            theta[theta==0]+=np.pi
            return theta

        node = self.entity('node')
        halfedge = self.entity('halfedge')

        hedge = self.ds.hedge
        hcell = self.ds.hcell
        cstart = self.ds.cellstart
        subdomain = self.ds.subdomain

        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']
        while True:
            NC = self.number_of_all_cells()
            NE = self.number_of_edges()

            end = halfedge[:, 0]
            start = halfedge[halfedge[:, 4], 0]
            vector = node[end] - node[start]#所有边的方向
            vectornex = vector[halfedge[:, 2]]#所有边的下一个边的方向

            angle0 = angle(vectornex, -vector)#所有边的反方向与下一个边的角度
            badHEdge, = np.where(angle0 > np.pi*17/18)#夹角大于170度的半边
            badCell, idx= np.unique(halfedge[badHEdge, 1], return_index=True)#每个单元每次只处理中的一个
            badHEdge = badHEdge[idx]#现在坏半边的单元都不同
            badHEdge = badHEdge[halfedge[badHEdge, 1]>cstart-1]
            badNode = halfedge[badHEdge, 0]
            NE1 = len(badHEdge)

            nex = halfedge[badHEdge, 2]
            pre = halfedge[badHEdge, 3]
            vectorBad = vector[badHEdge]#坏半边的方向
            vectorBadnex = vector[nex]#坏半边的下一个半边的方向

            anglenex = angle(vectorBadnex, -vectorBad)#坏边的夹角
            anglecur = anglenex/2#当前方向与角平分线的夹角
            angle_err_min = anglenex/2#与角平分线夹角的最小值
            goodHEdge = np.zeros(NE1, dtype=np.int_)#最小夹角的边
            isNotOK = np.ones(NE1, dtype = np.bool_)#每个单元的循环情况
            nex = halfedge[nex, 2]#从下下一个边开始
            while isNotOK.any():
                vectornex = node[halfedge[nex, 0]] - node[badNode]
                anglecur[isNotOK] = angle(vectorBadnex[isNotOK], vectornex[isNotOK])
                angle_err = abs(anglecur - anglenex/2)
                goodHEdge[angle_err<angle_err_min] = nex[angle_err<angle_err_min]#与角平分线夹角小于做小夹角的边做goodHEdge.
                angle_err_min[angle_err<angle_err_min] = angle_err[angle_err<angle_err_min]#更新最小角
                nex = halfedge[nex, 2]
                isNotOK[nex==pre] = False#循环到坏边的上上一个边结束
            halfedgeNew = halfedge.increase_size(NE1*2)
            halfedgeNew[:NE1, 0] = halfedge[goodHEdge, 0].copy()
            halfedgeNew[:NE1, 1] = halfedge[badHEdge, 1].copy()
            halfedgeNew[:NE1, 2] = halfedge[goodHEdge, 2].copy()
            halfedgeNew[:NE1, 3] = badHEdge.copy()
            halfedgeNew[:NE1, 4] = np.arange(NE*2+NE1, NE*2+NE1*2)

            halfedgeNew[NE1:, 0] = halfedge[badHEdge, 0].copy()
            halfedgeNew[NE1:, 1] = np.arange(NC, NC+NE1)
            halfedgeNew[NE1:, 2] = halfedge[badHEdge, 2].copy()
            halfedgeNew[NE1:, 3] = goodHEdge.copy()
            halfedgeNew[NE1:, 4] = np.arange(NE*2, NE*2+NE1)

            halfedge[halfedge[goodHEdge, 2], 3] = np.arange(NE*2, NE*2+NE1)
            halfedge[halfedge[badHEdge, 2], 3] = np.arange(NE*2+NE1, NE*2+NE1*2)
            halfedge[badHEdge, 2] = np.arange(NE*2, NE*2+NE1)
            halfedge[goodHEdge, 2] = np.arange(NE*2+NE1, NE*2+NE1*2)
            isNotOK = np.ones(NE1, dtype=np.bool_)
            nex = halfedge[-NE1:, 2]
            while isNotOK.any():
                halfedge[nex[isNotOK], 1] = np.arange(NC, NC+NE1)[isNotOK]
                nex = halfedge[nex, 2]
                flag = (nex==np.arange(NE*2+NE1, NE*2+NE1*2)) & isNotOK
                isNotOK[flag] = False

            #增加主半边
            hedge.extend(np.arange(NE*2, NE*2+NE1))

            #更新subdomain
            subdomainNew = subdomain.increase_size(NE1)
            subdomainNew[:] = subdomain[halfedge[badHEdge, 1]]

            #更新起始边
            hcell.increase_size(NE1)
            hcell[halfedge[:, 1]] = range(len(halfedge)) # 的编号

            #单元层
            clevelNew = clevel.increase_size(NE1)
            clevelNew[:] = clevel[halfedge[badHEdge, 1]]

            #半边层
            hlevelNew = hlevel.increase_size(NE1*2)
            hlevelNew[:NE1] = hlevel[goodHEdge]
            hlevelNew[NE1:] = hlevel[badHEdge]

            self.ds.NC = (subdomain[:]>0).sum()
            self.ds.NE = halfedge.size//2
            if len(badHEdge)==0:
                break

    def edge_normal(self, index=np.s_[:]):
        """
        @brief 计算二维网格中每条边上单位法线
        """
        assert self.geo_dimension() == 2
        v = self.edge_tangent(index=index)
        w = np.array([(0, -1), (1, 0)])
        return v @ w

    def edge_unit_normal(self, index=np.s_[:]):
        """
        @brief 计算二维网格中每条边上单位法线
        """
        assert self.geo_dimension() == 2
        v = self.edge_unit_tangent(index=index)
        w = np.array([(0, -1), (1, 0)])
        return v @ w

    def location(self, points):
        halfedge = self.entity('halfedge')
        node = self.entity('node')
        hcell = self.ds.hcell

        NP = points.shape[0]
        cell = np.ones(NP, dtype=self.itype)
        isNotOK = np.ones(NP, dtype=np.bool_)
        while isNotOK.any():
            cell2hedge = np.c_[hcell[cell],
                    halfedge[hcell[cell], 2], halfedge[hcell[cell], 3]].T
            cell2vector = node[halfedge[cell2hedge, 0]] - node[halfedge[
                halfedge[cell2hedge, 3], 0]]
            vector = x - node[halfedge[halfedge[cell2hedge, 3], 0]]
            area = np.cross(cell2vector, vector)
            idx, jdx = np.where(area<0)
            isNotOK[:] = False
            isNotOK[jdx]=True
            cell[jdx] = halfedge[halfedge[cell2hedge[idx, jdx], 4], 1]
        return cell

    def line_to_cell(self, point, segment):
        halfedge = self.entity('halfedge')
        node = self.entity('node')
        hcell = self.ds.hcell
        NC = self.ds.number_of_all_cells()

        x = point[segment[:, 0]]
        y = point[segment[:, 1]]

        start = self.node_to_cell(x)
        end = self.node_to_cell(y)
        isCrossCell = np.zeros(NC, dtype=np.bool_)
        isCrossCell[start] = True
        vector = x-y

        while len(start)>0:
            cell2hedge = np.c_[hcell[start],
                    halfedge[hcell[start], 2], halfedge[hcell[start], 3]].T
            cell2vector = node[halfedge[cell2hedge, 0]] - y
            area = np.cross(cell2vector, vector)
            area0 = np.r_['0', np.array([area[-1]]), area[:-1]]
            flag = ((area>=0)&(area0<0)).T
            start = halfedge[halfedge[cell2hedge.T[flag], 4], 1]
            isCrossCell[start] = True
            flag = (start==end)
            start = start[~flag]
            end = end[~flag]
            y = y[~flag]
            vector = vector[~flag]
        return isCrossCell, np.where(isCrossCell)

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

    def edge_bc_to_point(self, bcs, index=np.s_[:]):
        """
        @brief 重心坐标积分点转化为网格边上笛卡尔坐标点
        """
        node = self.entity('node')
        edge = self.entity('edge')[index]
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        return ps

    def bc_to_point(self, bc, index=np.s_[:]):
        """
        @brief 
        """
        node = self.entity('node')
        if self.ds.NV == 3:
            TD = bc.shape[-1] - 1 # bc.shape == (NQ, TD+1)
            node = self.entity('node')
            entity = self.entity(etype=TD)[index] # default  cell
            p = np.einsum('...j, ijk->...ik', bc, node[entity])
        elif self.ds.NV == 4:
            if isinstance(bc, tuple):
                assert len(bc) == 2
                cell = self.entity('cell')[index]
                bc0 = bc[0] # (NQ0, 2)
                bc1 = bc[1] # (NQ1, 2)
                bc = np.einsum('im, jn->ijmn', bc0, bc1).reshape(-1, 4) # (NQ0, NQ1, 2, 2)
                p = np.einsum('...j, cjk->...ck', bc, node[cell[:, [0, 3, 1, 2]]]) # (NQ, NC, 2)
            else:
                edge = self.entity('edge')[index]
                p = np.einsum('...j, ejk->...ek', bc, node[edge]) # (NQ, NE, 2)
        elif self.ds.NV is None:
            TD = bc.shape[-1] - 1 # bc.shape == (NQ, TD+1)
            node = self.entity('node')
            entity = self.entity(etype=TD)[index] # default  cell
            p = np.einsum('...j, ijk->...ik', bc, node[entity])
        return p

    def entity(self, etype=2, index=np.s_[:]):
        if etype in {'cell', 2}:
            return self.ds.cell_to_node()
        elif etype in {'edge', 'face', 1}:
            return self.ds.edge_to_node()[index]
        elif etype in {'halfedge'}:
            return self.ds.halfedge # DynamicArray
        elif etype in {'node', 0}:
            return self.node # DynamicArrray
        else:
            raise ValueError("`entitytype` is wrong!")

    def geo_dimension(self):
        return self.node.shape[1]
    
    def entity_measure(self, etype=2, index=np.s_[:]):
        if etype in {'cell', 2}:
            return self.cell_area(index=index)
        elif etype in {'edge', 'face', 1}:
            return self.halfedge_length()[self.ds.hedge][index]
        elif etype in {'node', 0}:
            return 0
        else:
            raise ValueError(f"Invalid entity type '{etype}'.")

    def entity_barycenter(self, etype='cell', index=np.s_[:]):
        node = self.entity('node')
        GD = self.geo_dimension()
        if etype in {'cell', 2}:
            # 这里是单元顶点坐标的平均位置
            cell2node = self.ds.cell_to_node(return_sparse=True)
            if self.ds.NV is None:
                NV = self.ds.number_of_vertices_of_cells()
                bc = cell2node@node/NV[:, None]
            else:
                bc = cell2node@node/self.ds.NV
        elif etype in {'edge', 'face', 1}:
            edge = self.ds.edge_to_node()
            bc = np.sum(node[edge, :], axis=1).reshape(-1, GD)/edge.shape[1]
        elif etype in {'node', 0}:
            bc = node
        elif etype in {'halfedge'}:
            halfedge = self.entity('halfedge')
            bc = 0.5*(node[halfedge[index, 0]] + node[halfedge[halfedge[index,
                4], 0]])
        return bc

    def node_normal(self):
        node = self.node
        cell = self.entity('cell')
        if isinstance(cell, tuple):
            cell, cellLocation = cell
            idx1 = np.zeros(cell.shape[0], dtype=np.int_)
            idx2 = np.zeros(cell.shape[0], dtype=np.int_)

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

    def cell_area(self, index=np.s_[:]):
        """

        Notes
        -----
        计算单元的面积
        """
        if self.ds.NV in {None, 4}:
            NC = self.number_of_cells()
            node = self.entity('node')

            halfedge = self.ds.halfedge # DynamicArray
            cstart = self.ds.cellstart

            flag = halfedge[:, 1] >= cstart
            e0 = halfedge[halfedge[flag, 3], 0]
            e1 = halfedge[flag, 0]

            w = np.array([[0, -1], [1, 0]], dtype=np.int_)
            v = (node[e1] - node[e0])@w
            val = np.sum(v*node[e0], axis=1)

            a = np.zeros(NC, dtype=self.ftype)
            np.add.at(a, halfedge[flag, 1] - cstart, val)
            a /=2
            return a
        elif self.ds.NV == 3:
            node = self.entity('node')
            cell = self.entity('cell')
            GD = self.geo_dimension()
            v1 = node[cell[index, 1], :] - node[cell[index, 0], :]
            v2 = node[cell[index, 2], :] - node[cell[index, 0], :]
            nv = np.cross(v1, v2)
            if GD == 2:
                a = nv/2.0
            elif GD == 3:
                a = np.sqrt(np.square(nv).sum(axis=1))/2.0
            return a

    def all_cell_area(self):
        cstart = self.ds.cellstart
        area = self.cell_area()
        area = np.r_[np.ones(cstart, dtype=np.float_), area]
        return np.abs(area)

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
            w = np.array([[0, -1], [1, 0]], dtype=np.int_)
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
            w = np.array([[0, -1], [1, 0]], dtype=np.int_)
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



    def mark_halfedge(self, isMarkedCell, method='poly'):
        cstart = self.ds.cellstart
        clevel = self.celldata['level'] # 注意这里是所有的单元的层信息
        nlevel = self.nodedata['level']
        hlevel = self.halfedgedata['level']
        halfedge = self.entity('halfedge')
        if method == 'poly':
            isMarkedHEdge = clevel[halfedge[:, 1]]>=clevel[halfedge[halfedge[:, 4], 1]]
            isMarkedHEdge = isMarkedHEdge|(halfedge[halfedge[:, 4], 1]<cstart)
            isMarkedHEdge = isMarkedHEdge & isMarkedCell[halfedge[:, 1]]
        elif method == 'quad':
            color = self.hedgecolor['color']
            isRedHEdge = color == 0
            isGreenHEdge = color == 1
            isOtherHEdge = (color == 2)|(color == 3)

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
            while True:
                flag0 = isGreenHEdge & ~isMarkedHEdge & isMarkedHEdge[halfedge[:, 3]]
                flag1 = isRedHEdge & ~isMarkedHEdge & isMarkedHEdge[halfedge[:, 2]]
                flag2 = isOtherHEdge & ~isMarkedHEdge & (isMarkedHEdge[halfedge[:,
                    2]] | isMarkedHEdge[halfedge[:, 3]])
                flag3 = isMarkedHEdge[halfedge[:, 4]] & ~isMarkedHEdge
                flag = flag0 | flag1 | flag2 | flag3

                isMarkedHEdge[flag] = True
                if (~flag).all():
                    break
        elif method == 'rg':
            color = self.hedgecolor
            isRedHEdge = color == 0
            isGreenHEdge = color == 1
            isOtherHEdge = (color == 2)|(color == 3)
            halfedge = self.ds.halfedge
            cstart = self.ds.cellstart
            isOutHEdge = halfedge[:, 1] <cstart

            # 当前半边的层标记小于等于所属单元的层标记
            flag0 = (hlevel - clevel[halfedge[:, 1]]) <= 0
            # 前一半边的层标记小于等于所属单元的层标记 
            pre = halfedge[:, 3]
            flag1 = (hlevel[pre] - clevel[halfedge[:, 1]]) <= 0
            # 标记加密的半边
            isMarkedHEdge = isMarkedCell[halfedge[:, 1]] & flag0 & flag1 & (~isGreenHEdge)
            while True:
                flag = ~isMarkedCell[halfedge[:, 1]] & isMarkedHEdge & (~isRedHEdge)
                isMarkedCell[halfedge[flag, 1]] = True

                flag0 = isMarkedCell[halfedge[:, 1]] & (~isGreenHEdge) & ~isOutHEdge & ~isMarkedHEdge
                flag1 = isMarkedHEdge[halfedge[:, 2]] & isMarkedHEdge[
                        halfedge[:, 3]] & isRedHEdge & ~isOutHEdge & ~isMarkedHEdge
                flag2 = (isMarkedHEdge[halfedge[:, 2]] | isMarkedHEdge[
                        halfedge[:, 3]]) & isOtherHEdge & ~isMarkedHEdge
                flag3 = isMarkedHEdge[halfedge[:, 4]] & ~isMarkedHEdge
                flag = flag0 | flag1 | flag2 | flag3

                isMarkedHEdge[flag] = True
                if (~flag).all():
                    break
        elif method == 'nvb':
            color = self.hedgecolor
            isRedHEdge = color == 1
            isBlueHEdge = color == 0
            halfedge = self.entity('halfedge')

            # 标记加密的半边
            isMarkedHEdge = isMarkedCell[halfedge[:, 1]] & isRedHEdge 
            while True:
                flag0 = isRedHEdge & (isMarkedHEdge[halfedge[:,
                    2]]|isMarkedHEdge[halfedge[:, 3]]) & ~isMarkedHEdge
                flag1 = isMarkedHEdge[halfedge[:, 4]] & ~isMarkedHEdge
                flag = flag0 | flag1

                isMarkedHEdge[flag] = True
                if (~flag).all():
                    break
        return isMarkedHEdge

    def refine_halfedge(self, isMarkedHEdge, newnode=None):

        NN = self.number_of_nodes()
        NE = self.number_of_edges()

        hlevel = self.halfedgedata['level']
        halfedge = self.entity('halfedge')
        node = self.entity('node')
        hedge = self.ds.hedge
        subdomain = self.ds.subdomain
        isMainHEdge = self.ds.main_halfedge_flag()

        #新点到新半边的映射
        self.newnode2edge, = np.where(isMarkedHEdge[hedge])

        # 即是主半边, 也是标记加密的半边
        flag0 = isMarkedHEdge & isMainHEdge
        idx = halfedge[flag0, 4]
        NE1 = flag0.sum()
        newNode = node.increase_size(NE1)
        if newnode is None:
            newNode[:] = (node[halfedge[flag0, 0]] + node[halfedge[idx, 0]])/2
        elif isinstance(newnode, np.ndarray):
            newNode[:] = newnode

        edge2NewNode = np.zeros(NE*2, dtype=np.int_)
        edge2NewNode[flag0] = np.arange(NE1)+NN
        edge2NewNode[idx] = np.arange(NE1)+NN

        #细分边
        newHalfedge = halfedge.increase_size(2*NE1)

        current, = np.where(isMarkedHEdge)
        pre = halfedge[current, 3]
        opp = halfedge[current, 4]

        halfedge[current, 3] = np.arange(NE*2, NE*2+NE1*2)
        halfedge[current, 4] = halfedge[opp, 3]
        halfedge[pre, 2] = np.arange(NE*2, NE*2+NE1*2)

        newHalfedge[:, 0] = edge2NewNode[isMarkedHEdge]
        newHalfedge[:, 1] = halfedge[:NE*2][isMarkedHEdge, 1]
        newHalfedge[:, 2] = current
        newHalfedge[:, 3] = pre
        newHalfedge[:, 4] = opp

        #更新主半边
        tmp, = np.where(flag0)
        newHedge = hedge.increase_size(NE1)
        newHedge[:] = halfedge[tmp, 3]

        #修改半边层
        hlevel[isMarkedHEdge] +=1
        hlevel.extend(np.zeros(2*NE1, dtype=np.int_))

        self.ds.NE = NE + NE1
        self.ds.NN = self.node.size
        return NE1

    def coarsen_halfedge(self, isMarkedHEdge):

        NN = self.number_of_nodes()
        NE = self.number_of_edges()

        hlevel = self.halfedgedata['level']
        halfedge = self.entity('halfedge')
        node = self.entity('node')
        hedge = self.ds.hedge
        hcell = self.ds.hcell
        subdomain = self.ds.subdomain
        isMainHEdge = self.ds.main_halfedge_flag()

        nex = halfedge[isMarkedHEdge, 2]
        pre = halfedge[isMarkedHEdge, 3]
        opp = halfedge[isMarkedHEdge, 4]

        halfedge[pre, 2] = nex
        halfedge[nex, 3] = pre
        halfedge[nex, 4] = opp

        isRNode = np.zeros(NN, dtype=np.bool_)
        isRNode[halfedge[isMarkedHEdge, 0]] = True

        #记录保留点
        self.retainnode, = np.where(~isRNode)
        self.deletenode, = np.where(isRNode)
        self.deletenode2edge = nex[isMainHEdge[nex]]

        #更新节点
        node.adjust_size(isRNode)
        nn = (~isRNode).sum()
        nidxmap = np.arange(NN)
        nidxmap[~isRNode] = range(nn)
        halfedge[:, 0] = nidxmap[halfedge[:, 0]]

        # 更新半边层
        hlevel[nex]-=1
        hlevel.adjust_size(isMarkedHEdge)

        # 对半边重新编号
        ne = np.sum(~isMarkedHEdge)//2
        L = len(isMarkedHEdge)
        l = (~isMarkedHEdge).sum()
        idxmap = np.arange(L)
        idxmap[~isMarkedHEdge] = np.arange(l)
        halfedge[:, 2:] = idxmap[halfedge[:, 2:]]
        self.deletnode2edge = idxmap[self.deletenode2edge]

        # 更新halfedge
        halfedge.adjust_size(isMarkedHEdge)

        #更新起始边
        hcell[halfedge[:, 1]] = range(len(halfedge))

        # 更新主半边
        hedge.decrease_size(NE-ne)
        hedge[:], = np.where(isMainHEdge[~isMarkedHEdge])

        self.ds.NE = ne
        self.ds.NN = self.node.size

    def refine_cell(self, isMarked, isMarkedHEdge, method='quad',
            bc=None, options={}):

        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_all_cells()

        node = self.entity('node')
        halfedge = self.entity('halfedge')
        clevel = self.celldata['level']
        hlevel = self.halfedgedata['level']
        hedge = self.ds.hedge
        hcell = self.ds.hcell
        cstart = self.ds.cellstart
        subdomain = self.ds.subdomain

        NC1 = isMarkedHEdge.sum()
        NN1 = isMarked.sum()
        isMainHEdge = self.ds.main_halfedge_flag()

        if method=='quad':
            isMarkedCell = isMarked
            #生成新的节点
            if bc is None:
                bc = self.cell_barycenter()
                node.extend(bc[isMarkedCell[cstart:NC]])
            else:
                node.extend(bc[isMarkedCell[:NC]])
            cell2newNode = np.zeros(NC, dtype=np.int_)
            cell2newNode[isMarkedCell[:NC]] = np.arange(node.size)[-NN1:]

            #生成新的单元
            current, = np.where(isMarkedHEdge)
            pre = halfedge[current, 3]
            nex = halfedge[current, 2]

            cidx = halfedge[current, 1]
            halfedge[current, 1] = np.arange(NC, NC+NC1)
            halfedge[nex, 1] = np.arange(NC, NC+NC1)
            isNotOK = ~isMarkedHEdge[halfedge[nex, 2]]
            idx = nex.copy()
            while isNotOK.any():
                idx[isNotOK] = halfedge[nex[isNotOK], 2]
                flag = isMarkedHEdge[idx]
                nex[~flag] = idx[~flag]
                halfedge[nex, 1] = np.arange(NC, NC+NC1)
                isNotOK[flag]=False
            ppre = halfedge[halfedge[:, 3], 3]
            ppre[halfedge[nex, 2]] = current
            ppre = ppre[current]
            halfedge[current, 3] = np.arange(NE*2+NC1, NE*2+NC1*2)
            halfedge[pre, 2] = np.arange(NE*2, NE*2+NC1)

            halfedgeNew = halfedge.increase_size(NC1*2)
            halfedgeNew[:NC1, 0] = cell2newNode[cidx]
            halfedgeNew[:NC1, 1] = halfedge[pre, 1]
            halfedgeNew[:NC1, 2] = halfedge[ppre, 3]
            halfedgeNew[:NC1, 3] = pre
            halfedgeNew[:NC1, 4] = halfedge[current, 3]

            halfedgeNew[NC1:, 0] = halfedge[pre, 0]
            halfedgeNew[NC1:, 1] = halfedge[current, 1]
            halfedgeNew[NC1:, 2] = current
            halfedgeNew[NC1:, 3] = halfedge[nex, 2]
            halfedgeNew[NC1:, 4] = halfedge[pre, 2]

            #修改单元编号
            cidxmap = np.arange(NC+NC1)
            idx = np.unique(halfedge[:, 1])
            NC2 = idx.shape[0]
            cidxmap[idx] = np.arange(NC2)
            halfedge[:, 1] = cidxmap[halfedge[:, 1]]

            if 'numrefine' in options:
                num = options['numrefine']
                num = np.r_[num, np.zeros(NC1, dtype=np.int_)]
                num[-NC1:] = num[cidx]-1#current所属的单元加密次数-1

                #若不需要加密的单元加密了, 将他的子单元加密次数设为0
                flag = np.zeros(len(num), dtype=np.bool_)
                flag[-NC1:] = True
                flag = flag & (num<0)
                num[flag] = 0

                options['numrefine'] = num[idx]

            if ('HB' in options) and (options['HB'] is not None):
                isNonMarkedCell = ~isMarkedCell
                flag0 = isNonMarkedCell[cstart:]
                flag1 = isMarkedCell[cstart:]
                NHB0 = flag0.sum()
                NHE = len(halfedge)
                NHB = NHB0 + NHE
                HB = np.zeros((NHB, 2), dtype=np.int_)
                HB[:, 0] = range(NHB)
                HB[0:NHB0, 1] = np.arange(len(flag0))[flag0]
                HB[NHB0:,  1] = cellidx - cellstart
                HB0 = HB.copy()
                HB0[:, 1] = options['HB'][HB0[:,1], 1]
                options['HB'] = HB0

            #增加主半边
            hedge.extend(np.arange(NE*2, NE*2+NC1))

            #更新subdomain
            subdomainNew = subdomain.adjust_size(isMarkedCell, int(NC1))
            subdomainNew[:] = subdomain[cidx]

            #更新起始边
            hcell.increase_size(NC2-NC)
            hcell[halfedge[:, 1]] = np.arange(len(halfedge)) # 的编号

            #单元层
            clevel1 = clevel[cidx]
            clevelNew = clevel.adjust_size(isMarkedCell, int(NC1))
            clevelNew[:] = clevel1+1

            #半边层
            hlevel.extend(np.zeros(NC1*2, dtype=np.int_))

            self.ds.NN = self.node.size
            self.ds.NC = (subdomain[:]>0).sum()
            self.ds.NE = halfedge.size//2

        elif method=='tri':
            isMarkedCell = isMarked
            NN1 = isMarkedCell.sum()

            current, = np.where(isMarkedHEdge)
            pre = halfedge[current, 3]
            ppre = halfedge[pre, 3]
            nex = halfedge[current, 2]
            nnex = halfedge[nex, 2]

            cidx = halfedge[current, 1]
            halfedge[nnex, 3] = np.arange(NE*2+NC1, NE*2+NC1*2)
            halfedge[pre, 2] = np.arange(NE*2+NC1, NE*2+NC1*2)
            halfedge[current, 1] = np.arange(NC, NC+NC1)
            halfedge[current, 3] = np.arange(NE*2, NE*2+NC1)
            halfedge[nex, 1] = np.arange(NC, NC+NC1)
            halfedge[nex, 2] = np.arange(NE*2, NE*2+NC1)

            halfedgeNew = halfedge.increase_size(NC1*2)
            halfedgeNew[:NC1, 0] = halfedge[pre, 0]
            halfedgeNew[:NC1, 1] = np.arange(NC, NC+NC1)
            halfedgeNew[:NC1, 2] = current
            halfedgeNew[:NC1, 3] = nex
            halfedgeNew[:NC1, 4] = np.arange(NE*2+NC1, NE*2+NC1*2)

            halfedgeNew[NC1:, 0] = halfedge[nex, 0]
            halfedgeNew[NC1:, 1] = cidx
            halfedgeNew[NC1:, 2] = nnex
            halfedgeNew[NC1:, 3] = pre
            halfedgeNew[NC1:, 4] = np.arange(NE*2, NE*2+NC1)

            if ('HB' in options) and (options['HB'] is not None):
                NC0 = NC-cstart
                HB0 = np.zeros([NC0+NC1, 2], dtype=np.int_)
                HB0[:, 0] = np.arange(NC0+NC1)
                HB0[:NC0, 1] = np.arange(NC0)
                HB0[NC0:, 1] = cidx-cstart
                HB = options['HB']
                HB0[:, 1] = HB[HB0[:, 1], 1]
                options['HB'] = HB0

            flag = isMarkedHEdge[halfedgeNew[NC1:, 2]]
            current = np.where(flag)[0]+NC1
            nexpre = halfedge[halfedgeNew[current, 2], 3]
            prenex = halfedge[halfedgeNew[current, 3], 2]
            halfedgeNew[current, 2] = halfedge[nexpre, 4]
            halfedgeNew[current, 3] = halfedge[prenex, 4]

            if ('numrefine' in options) & (NC1>0):
                num = options['numrefine']
                num = np.r_[num, np.zeros(NC1)]

                #若加密次数小于等于0, 却被加密了, 子单元加密次数设为 0.
                num[cidx]-=1
                flag = np.zeros(len(num), dtype=np.bool_)
                flag[cidx] = True
                flag  = flag & (num<0)
                num[flag]=0

                num[-NC1:] = num[cidx]
                options['numrefine'] = num

            #增加主半边
            hedge.extend(np.arange(NE*2, NE*2+NC1))

            #更新subdomain
            subdomainNew = subdomain.increase_size(NC1)
            subdomainNew[:] = subdomain[cidx]

            #更新起始边
            hcell.increase_size(NC1)
            hcell[halfedge[:, 1]] = range(len(halfedge)) # 的编号

            #单元层
            clevelNew = clevel.increase_size(NC1)
            clevel[cidx] +=1
            clevelNew[:] = clevel[cidx]

            #半边层
            hlevel.extend(np.zeros(NC1*2, dtype=np.int_))

            self.ds.NN = self.node.size
            self.ds.NC = (subdomain[:]>0).sum()
            self.ds.NE = halfedge.size//2

        elif method=='quad_coordinateCell':
            # 标记的蓝色单元变成红色单元
            flag = isMarked
            NE1 = flag.sum()

            halfedgeNew = halfedge.increase_size(NE1*2)
            isMainHEdgeNew = np.zeros(NE1*2, dtype=np.bool_)

            current, = np.where(flag)
            pre = halfedge[current, 3]
            ppre = halfedge[pre, 3]
            opp = halfedge[current, 4]

            halfedge[current, 3] = ppre# 修改蓝色半边位置
            halfedge[current, 4] = np.arange(NE*2, NE*2+NE1)
            halfedge[ppre, 2] = current
            halfedge[pre, 1] = np.arange(NC, NC+NE1)
            halfedge[pre, 2] = halfedge[opp, 2]
            halfedge[pre, 3] = np.arange(NE*2, NE*2+NE1)
            halfedgeNew[:NE1, 0] = halfedge[ppre, 0]
            halfedgeNew[:NE1, 1] = np.arange(NC, NC+NE1)
            halfedgeNew[:NE1, 2] = pre
            halfedgeNew[:NE1, 3] = np.arange(NE*2+NE1, NE*2+NE1*2)
            halfedgeNew[:NE1, 4] = current
            isMainHEdgeNew[:NE1] = ~isMainHEdge[current]
            isMarkedHEdge[pre] = False

            current = opp.copy()
            nex = halfedge[current, 2]
            nnex = halfedge[nex, 2]
            opp = halfedge[current, 4]

            halfedge[current, 0] = halfedge[nex, 0]# 修改黄色半边位置
            halfedge[current, 2] = nnex
            halfedge[current, 4] = np.arange(NE*2+NE1, NE*2+NE1*2)
            halfedge[nnex, 3] = current
            halfedge[nex, 1] = np.arange(NC, NC+NE1)
            halfedge[nex, 2] = np.arange(NE*2+NE1, NE*2+NE1*2)
            halfedge[nex, 3] = pre
            halfedgeNew[NE1:, 0] = halfedge[opp, 0]
            halfedgeNew[NE1:, 1] = np.arange(NC, NC+NE1)
            halfedgeNew[NE1:, 2] = np.arange(NE*2, NE*2+NE1)
            halfedgeNew[NE1:, 3] = nex
            halfedgeNew[NE1:, 4] = current
            isMainHEdgeNew[NE1:] = ~isMainHEdge[current]
            isMarkedHEdge[nnex] = False

            if ('numrefine' in options) & (NE1>0):
                opp = halfedge[:, 4]
                num = options['numrefine']
                num0 = num[halfedge[opp[-NE1*2:-NE1], 1]]
                num1 = num[halfedge[opp[-NE1:], 1]]
                num2 = np.maximum(num0, num1)
                num = np.r_[num, num2]
                options['numrefine'] = num

            #半边层
            hlevel.extend(np.zeros(NE1*2, dtype=np.int_))

            #单元层
            clevel.extend(clevel[halfedge[current, 1]])

            #增加主半边
            newHedge = hedge.increase_size(NE1)
            newHedge[:] = np.where(isMainHEdgeNew)[0]+NE*2

            #更新subdomain
            subdomainNew = subdomain.increase_size(NE1)
            subdomainNew[:] = subdomain[halfedge[current, 1]]

            #更新起始边
            hcell.increase_size(NE1)
            hcell[halfedge[:, 1]] = range(len(halfedge))

            self.ds.NN = self.node.size
            self.ds.NC = (subdomain[:]>0).sum()
            self.ds.NE = halfedge.size//2

        elif method=='tri_coordinateCell':
            flag = isMarked
            NE1 = flag.sum()
            halfedgeNew = halfedge.increase_size(NE1*4)
            isMainHEdgeNew = np.zeros(NE1*4, dtype=np.bool_)

            current, = np.where(flag)
            pre = halfedge[current, 3]
            ppre = halfedge[pre, 3]
            pppre = halfedge[ppre, 3]
            opp = halfedge[current, 4]

            halfedge[current, 3] = ppre#修改蓝色半边位置
            halfedge[current, 4] = np.arange(NE*2, NE*2+NE1)
            halfedge[ppre, 2] = current
            halfedge[pre, 1] = np.arange(NC, NC+NE1)
            halfedge[pre, 2] = halfedge[opp, 2]
            halfedge[pre, 3] = np.arange(NE*2+NE1*2, NE*2+NE1*3)
            halfedgeNew[:NE1, 0] = halfedge[ppre, 0]
            halfedgeNew[:NE1, 1] = np.arange(NC+NE1, NC+NE1*2)
            halfedgeNew[:NE1, 2] = np.arange(NE*2+NE1*3, NE*2+NE1*4)
            halfedgeNew[:NE1, 3] = np.arange(NE*2+NE1, NE*2+NE1*2)
            halfedgeNew[:NE1, 4] = current
            halfedgeNew[NE1*2:NE1*3, 0] = halfedge[ppre, 0]
            halfedgeNew[NE1*2:NE1*3, 1] = np.arange(NC, NC+NE1)
            halfedgeNew[NE1*2:NE1*3, 2] = pre
            halfedgeNew[NE1*2:NE1*3, 3] = halfedge[opp, 2]
            halfedgeNew[NE1*2:NE1*3, 4] = np.arange(NE*2+NE1*3, NE*2+NE1*4)
            isMainHEdgeNew[:NE1] = ~isMainHEdge[current]
            isMarkedHEdge[pre] = False

            current = opp.copy()
            nex = halfedge[current, 2]
            nnex = halfedge[nex, 2]
            nnnex = halfedge[nnex, 2]
            opp = halfedge[current, 4]

            halfedge[current, 0] = halfedge[nex, 0]#修改黄色半边位置
            halfedge[current, 2] = nnex
            halfedge[current, 4] = np.arange(NE*2+NE1, NE*2+NE1*2)
            halfedge[nnex, 3] = current
            halfedge[nex, 1] = np.arange(NC, NC+NE1)
            halfedge[nex, 2] = np.arange(NE*2+NE1*2, NE*2+NE1*3)
            halfedge[nex, 3] = pre
            halfedgeNew[NE1:NE1*2, 0] = halfedge[opp, 0]
            halfedgeNew[NE1:NE1*2, 1] = np.arange(NC+NE1, NC+NE1*2)
            halfedgeNew[NE1:NE1*2, 2] = np.arange(NE*2, NE*2+NE1)
            halfedgeNew[NE1:NE1*2, 3] = np.arange(NE*2+NE1*3, NE*2+NE1*4)
            halfedgeNew[NE1:NE1*2, 4] = current
            halfedgeNew[NE1*3:NE1*4, 0] = halfedge[nex, 0]
            halfedgeNew[NE1*3:NE1*4, 1] = np.arange(NC+NE1, NC+NE1*2)
            halfedgeNew[NE1*3:NE1*4, 2] = np.arange(NE*2+NE1, NE*2+NE1*2)
            halfedgeNew[NE1*3:NE1*4, 3] = np.arange(NE*2, NE*2+NE1)
            halfedgeNew[NE1*3:NE1*4, 4] = np.arange(NE*2+NE1*2, NE*2+NE1*3)
            isMainHEdgeNew[NE1:NE1*2] = ~isMainHEdge[current]
            isMainHEdgeNew[NE1*3:NE1*4] = True
            isMarkedHEdge[nnex] = False

            #中心单元编号最小
            cidxmap = np.arange(NC+NE1*2)
            cidxmap[halfedge[current, 1]] = np.arange(NC+NE1, NC+NE1*2)
            cidxmap[NC+NE1:NC+NE1*2] = halfedge[current, 1]
            halfedge[:, 1] = cidxmap[halfedge[:, 1]]

            if 'numrefine' in options:
                num = options['numrefine']
                num0 = num[halfedge[halfedge[current, 4], 1]].copy()
                num1 = num[halfedge[opp, 1]]
                num2 = np.maximum(num0, num1)
                num[halfedge[halfedge[current, 4], 1]] = num2

                num = np.r_[num, num2, num0]
                options['numrefine'] = num

            color = np.r_[current, opp, nex, pre, ppre, nnex, pppre, nnnex]

            #半边层
            hlevel.extend(np.zeros(NE1*4, dtype=np.int_))

            #单元层
            clevel.extend(clevel[halfedge[opp, 1]])
            clevel.extend(clevel[halfedge[opp, 1]])

            #增加主半边
            newHedge = hedge.increase_size(NE1*2)
            newHedge[:] = np.where(isMainHEdgeNew)[0]+NE*2

            #更新subdomain
            subdomainNew = subdomain.increase_size(NE1*2)
            subdomainNew[:NE1] = subdomain[halfedge[opp, 1]]
            subdomainNew[NE1:] = subdomain[halfedge[opp, 1]]

            #更新起始边
            hcell.increase_size(NE1*2)
            hcell[halfedge[:, 1]] = range(len(halfedge)) # 的编号

            self.ds.NN = self.node.size
            self.ds.NC = (subdomain[:]>0).sum()
            self.ds.NE = halfedge.size//2
            return color

    def coarsen_cell(self, isMarkedCell, isMarked, method='quad', options={}):

        NC = self.number_of_all_cells()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        hcell = self.ds.hcell
        hedge = self.ds.hedge
        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']

        node = self.entity('node')
        halfedge = self.entity('halfedge')
        cstart = self.ds.cellstart
        subdomain = self.ds.subdomain
        isMainHEdge = self.ds.main_halfedge_flag()

        if method=='quad':
            isRNode = isMarked
            flag = isMarkedCell[halfedge[:, 1]]
            np.logical_and.at(isRNode, halfedge[:, 0], flag)
            nn = isRNode.sum()
            newNode = node[isRNode]

            # 重新标记要移除的单元
            isMarkedHEdge = isRNode[halfedge[:, 0]]
            isMarkedHEdge[halfedge[isMarkedHEdge, 4]] = True
            isMarkedCell = np.zeros(NC+nn, dtype=np.bool_)
            isMarkedCell[halfedge[isMarkedHEdge, 1]] = True
            nc = np.sum(~isMarkedCell[:NC])# 没有被标记的单元个数

            # 更新subdomain
            nsd = np.zeros(NN, dtype=self.itype)
            nsd[halfedge[:, 0]] = subdomain[halfedge[:, 1]]
            subdomain.adjust_size(isMarkedCell[:NC], nsd[isRNode])

            # 更新单元层
            ncl = np.zeros(NN, dtype=self.itype)
            ncl[halfedge[:, 0]] = clevel[halfedge[:, 1]]-1
            clevel.adjust_size(isMarkedCell[:NC], ncl[isRNode])

            # 更新单元编号
            cidxmap = np.arange(NC)
            nidxmap = np.arange(NN)
            nidxmap[isRNode] = range(nc, nc+nn)
            isRHEdge = isRNode[halfedge[:, 0]]
            #修改标记单元的编号
            cidxmap[halfedge[isRHEdge, 1]] = nidxmap[halfedge[isRHEdge, 0]]
            #修改未标记单元的编号
            cidxmap[~isMarkedCell[:NC]] = np.arange(nc)
            halfedge[:, 1] = cidxmap[halfedge[:, 1]]

            if 'numrefine' in options:
                num0 = np.zeros(nc+nn)-10000
                num = options['numrefine']
                num[isMarkedCell[:NC]]+=1
                np.maximum.at(num0, cidxmap, num)
                options['numrefine'] = num0

            # 修改保留半边的下一个边, 上一个边
            nex = halfedge[:, 2]
            pre = halfedge[:, 3]
            opp = halfedge[:, 4]
            flag1 = isMarkedHEdge[nex]
            halfedge[flag1, 2] = nex[opp[nex[flag1]]]
            flag2 = isMarkedHEdge[pre]
            halfedge[flag2, 3] = pre[opp[pre[flag2]]]

            # 对半边重新编号
            self.adjust_number(isMarkedHEdge, method='halfedge')

            #更新节点编号
            self.adjust_number(isRNode, method='node')

            # 更新halfedge
            halfedge.adjust_size(isMarkedHEdge)

            #更新起始边
            hcell.decrease_size(NC-nn-nc)
            hcell[halfedge[:, 1]] = range(len(halfedge))

            # 更新主半边
            hedge.decrease_size(isMarkedHEdge.sum()//2)
            hedge[:], = np.where(isMainHEdge[~isMarkedHEdge])

            # 更新半边层
            hlevel.adjust_size(isMarkedHEdge)

            #更新节点
            node.adjust_size(isRNode)

            self.ds.NN = self.node.size
            self.ds.NC = (subdomain[:]>0).sum()
            self.ds.NE = halfedge.size//2
            return isMarkedHEdge, isRNode, newNode

        elif method=='tri':
            isMarkedHEdge = isMarked

            flag = (halfedge[:, 1]<halfedge[halfedge[:, 4], 1]) & isMarkedHEdge
            isRCell = np.zeros(NC, dtype=np.bool_)
            isRCell[halfedge[flag, 1]] = True

            #修改下一个边和上一个边
            for i in range(2):
                nex = halfedge[:, 2]
                pre = halfedge[:, 3]
                opp = halfedge[:, 4]
                flag1 = isMarkedHEdge[nex]
                halfedge[flag1, 2] = nex[opp[nex[flag1]]]
                flag2 = isMarkedHEdge[pre]
                halfedge[flag2, 3] = pre[opp[pre[flag2]]]

            #修改单元
            flag = isRCell[halfedge[: ,1]] & isMarkedHEdge

            nc = isRCell.sum()
            opp = halfedge[:, 4]
            cidxmap = np.arange(NC)
            cidxmap[halfedge[opp[flag], 1]] = cidxmap[halfedge[flag, 1]]
            halfedge[:, 1] = cidxmap[halfedge[:, 1]]

            cell0 = np.unique(halfedge[:, 1])
            cidxmap0 = np.arange(NC)
            cidxmap0[cell0] = np.arange(len(cell0))
            halfedge[:, 1] = cidxmap0[halfedge[:, 1]]

            if ('HB' in options) and (options['HB'] is not None):
                HB = np.zeros((NC-cstart, 2), dtype=np.int_)
                HB[:, 0] = np.arange(NC-cstart)
                HB[:, 1] = cidxmap0[cidxmap[cstart:]]-cstart
                options['HB'] = HB

            if 'numrefine' in options:
                num = options['numrefine']
                np.maximum.at(num, cidxmap, num)
                num[halfedge[flag, 1]] += 1
                options['numrefine'] = num[cell0]

            #更新subdomain
            flag = cidxmap!=np.arange(NC)
            subdomain.adjust_size(flag)

            #更新单元层
            clevel[isRCell]-=1
            clevel.adjust_size(flag)

            #更新半边层
            hlevel.adjust_size(isMarkedHEdge)

            # 对半边重新编号
            self.adjust_number(isMarkedHEdge, method='halfedge')

            # 更新halfedge
            halfedge.adjust_size(isMarkedHEdge)

            #更新起始边
            hcell.decrease_size(NC-len(cell0))
            hcell[halfedge[:, 1]] = range(len(halfedge))

            # 更新主半边
            hedge.decrease_size(isMarkedHEdge.sum()//2)
            hedge[:], = np.where(isMainHEdge[~isMarkedHEdge])

            self.ds.NN = self.node.size
            self.ds.NC = (subdomain[:]>0).sum()
            self.ds.NE = halfedge.size//2
            return isMarkedHEdge

    def refine_poly(self, isMarkedCell=None, options={'disp': True}):
        NC = self.ds.number_of_all_cells()
        if isMarkedCell is None:
            isMarkedCell = np.ones(NC, dtype=np.bool_)

        cstart = self.ds.cellstart
        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']
        halfedge = self.entity('halfedge')
        # 所有单元的重心, 包括外部无界区域和区域中的洞区域
        bc = self.cell_barycenter(return_all=True)
        isMarkedCell[:cstart]=False

        isMarkedHEdge = self.mark_halfedge(isMarkedCell, method='poly')

        opp = halfedge[:, 4]
        cidx = halfedge[:, 1]
        flag = (clevel[cidx]+hlevel[:]+1==clevel[cidx[opp]]) & isMarkedCell[cidx]
        tmp, = np.where(flag)
        tmp2, = np.where(isMarkedHEdge)

        isMarkedHEdge[halfedge[isMarkedHEdge, 4]] = True
        NE1 = self.refine_halfedge(isMarkedHEdge)
        flag = np.zeros(len(halfedge), dtype=np.bool_)
        flag[tmp2]=True
        flag[halfedge[tmp, 2]]=True

        self.refine_cell(isMarkedCell, flag, method='quad', options=options)

    def coarsen_poly(self, isMarkedCell, i=0, options={'disp': True}):

        NC = self.number_of_all_cells()
        NN = self.number_of_nodes().copy()
        NE = self.number_of_edges()
        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']

        node = self.entity('node')
        halfedge = self.entity('halfedge')
        cstart = self.ds.cellstart
        subdomain = self.ds.subdomain
        isMainHEdge = self.ds.main_halfedge_flag()

        # 可以移除的网格节点
        isRNode = np.ones(NN, dtype=np.bool_)
        flag = (hlevel == hlevel[halfedge[:, 4]])
        np.logical_and.at(isRNode, halfedge[:, 0], flag)
        flag = (clevel[halfedge[:, 1]]>0)
        np.logical_and.at(isRNode, halfedge[:, 0], flag)

        #粗化单元
        self.coarsen_cell(isMarkedCell, isRNode)
        nex = halfedge[:, 2]
        opp = halfedge[:, 4]
        flag = (opp[nex[opp[nex]]] == np.arange(len(halfedge)))
        flag = flag & (hlevel[:]==0) & (hlevel[halfedge[:, 2]]>0)
        self.coarsen_halfedge(flag)

    def refine_quad(self, isMarkedCell, options={}):

        NC = self.number_of_all_cells()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        halfedge = self.ds.halfedge

        if self.hedgecolor is None:
            color = 3*np.ones(NE*2, dtype = np.int_)
            color[1]=1
            while (color==3).any():
                red = color == 1
                gre = color == 0
                color[halfedge[red][:, [2,3,4]]] = 0
                color[halfedge[gre][:, [2,3,4]]] = 1
            colorlevel = ((color==1) | (color==2)).astype(np.int_)
            self.hedgecolor = {'color':color, 'level':colorlevel}

        color = self.hedgecolor['color']
        colorlevel = self.hedgecolor['level']

        node = self.entity('node')
        cstart = self.ds.cellstart
        subdomain = self.ds.subdomain
        hedge = self.ds.hedge
        hcell = self.ds.hcell

        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']
        isBlueHedge = color==2
        isYellowHedge = color==3

        # 得到所有单元的中心, 包括外部无界区域和区域中的洞区域
        self.ds.NV = 4
        bc = np.r_[np.zeros([cstart, 2]), node[self.ds.cell_to_node()].sum(axis = 1)/4]
        self.ds.NV = None

        nex = halfedge[isBlueHedge, 2]
        nnex = halfedge[nex, 2]
        pre = halfedge[isBlueHedge, 3]
        bc[halfedge[isBlueHedge, 1]] +=(node[halfedge[nnex, 0]] -
                node[halfedge[pre, 0]])/8# 修改蓝色半边对应单元的中心

        nex = halfedge[isYellowHedge, 2]
        nnex = halfedge[nex, 2]
        bc[halfedge[isYellowHedge, 1]] +=(node[halfedge[nex, 0]] -
                node[halfedge[isYellowHedge, 0]])/8# 修改黄色半边对应单元的中心

        # 得到标记半边并加密
        isMarkedCell[:cstart] = False
        isMarkedHEdge0 = self.mark_halfedge(isMarkedCell, method='quad')
        isMarkedHEdge = isMarkedHEdge0 & ((color==0)|(color==1))

        isMarkedCell[halfedge[isMarkedHEdge, 1]] = True
        isMarkedCell[:cstart] = False
        NN1 = self.refine_halfedge(isMarkedHEdge)

        # 改变半边的颜色
        color = np.r_[color, np.zeros(NN1*2, dtype=np.int_)]
        color[halfedge[NE*2:, 4]] = 1
        colorlevel = np.r_[colorlevel, np.zeros(NN1*2, dtype=np.int_)]
        colorlevel[halfedge[NE*2:, 4]] += 1
        NNE = NE+NN1

        # 标记的蓝色单元变成红色单元
        flag = isBlueHedge & isMarkedHEdge0
        NE1 = flag.sum()
        self.refine_cell(flag, isMarkedHEdge, method='quad_coordinateCell', options=options)

        # 修改半边颜色
        color = np.r_[color, np.zeros(NE1*2, dtype=np.int_)]
        color[NNE*2:NNE*2+NE1] = 0
        color[NNE*2+NE1:NNE*2+NE1*2] = 1
        color[halfedge[NNE*2:, 4]] = (color[NNE*2:]+1)%2
        colorlevel = np.r_[colorlevel, np.zeros(NE1*2, dtype=np.int_)]
        colorlevel[NNE*2+NE1:NNE*2+NE1*2] +=1

        #生成新单元
        NV = self.ds.number_of_vertices_of_all_cells()
        isNewCell = (NV==6)|(NV==8)
        bc = np.r_[bc, np.zeros([NE1, 2], dtype=np.float_)]
        flag = (color==1) & isNewCell[halfedge[:, 1]]
        self.refine_cell(isNewCell, flag, method='quad', bc=bc, options=options )

        #修改半边颜色
        NC1 = flag.sum()
        isMarkedHEdge = np.r_[isMarkedHEdge, np.zeros(NN1*2+NE1*2, dtype=np.bool_)]
        tmp = np.where(~isMarkedHEdge & flag)[0]
        color = np.r_[color, np.zeros(NC1*2, dtype = np.int_)]
        color[-NC1*2:-NC1] = 1
        color[halfedge[tmp, 3]]=3
        color[halfedge[halfedge[tmp, 3], 4]] = 2
        colorlevel = np.r_[colorlevel, np.zeros(NC1*2, dtype = np.int_)]
        colorlevel[-NC1*2:-NC1] += 1
        self.hedgecolor['color'] = color
        self.hedgecolor['level'] = colorlevel

    def coarsen_quad(self, isMarkedCell, options={'disp': True}):

        NC = self.number_of_all_cells()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']

        color = self.hedgecolor['color']
        colorlevel = self.hedgecolor['level']
        halfedge = self.ds.halfedge
        cstart = self.ds.cellstart
        subdomain = self.ds.subdomain
        isMainHEdge = self.ds.main_halfedge_flag()

        # 可以移除的网格节点
        isRNode = np.ones(NN, dtype=np.bool_)
        flag = (hlevel == hlevel[halfedge[:, 4]])
        np.logical_and.at(isRNode, halfedge[:, 0], flag)
        flag = (clevel[halfedge[:, 1]]>0)
        np.logical_and.at(isRNode, halfedge[:, 0], flag)

        #粗化单元
        isMarkedHEdge, isRNode, newNode = self.coarsen_cell(isMarkedCell, isRNode)
        nn = isRNode.sum()

        #更新颜色
        color = color[~isMarkedHEdge]
        colorlevel = colorlevel[~isMarkedHEdge]

        # 标记进一步要移除的半边
        isMarkedNode = np.ones((~isRNode).sum(), dtype=np.bool_)
        pre = halfedge[:, 3]
        opp = halfedge[:, 4]
        flag0 = opp[pre[opp[pre]]] == np.arange(len(halfedge))
        np.logical_and.at(isMarkedNode, halfedge[:, 0], flag0 & (colorlevel==1))

        flag = isMarkedNode[halfedge[:, 0]] & (hlevel!=hlevel[halfedge[:, 4]])#可以变成红色半边的半边
        flag = flag[halfedge[:, 4]]
        flag[halfedge[halfedge[flag, 2], 4]] = True#要被移除的边

        #修改半边颜色
        colorlevel[halfedge[flag, 2]]-=1
        colorlevel = colorlevel[~flag]
        color = color[~flag]
        color[colorlevel==0]=0

        #删除边上的半边
        self.coarsen_halfedge(flag)

        #生成新节点, 生成新单元
        NV = self.ds.number_of_vertices_of_all_cells()
        isNewCell = (NV==6)|(NV==8)
        bc = np.zeros([len(NV), 2], dtype=np.float_)
        bc[isNewCell] = newNode[isNewCell[-nn:]]

        tmp = (hlevel == hlevel[halfedge[:, 4]]) & (NV==6)[halfedge[:, 1]]
        tmp, = np.where(tmp)

        flag = (color==1) & (isNewCell[halfedge[:, 1]])
        NC1 = flag.sum()
        self.refine_cell(isNewCell, flag, method='quad', bc=bc, options=options)#新节点新单元

        #修改半边颜色
        color = np.r_[color, np.zeros(NC1*2, dtype = np.int_)]
        color[-NC1*2:-NC1] = 1
        color[halfedge[tmp[color[tmp]==0], 2]]=2
        color[halfedge[tmp[color[tmp]==1], 3]]=3
        color[halfedge[color==2, 4]] = 3
        color[halfedge[color==3, 4]] = 2
        colorlevel = np.r_[colorlevel, np.zeros(NC1*2, dtype = np.int_)]
        colorlevel[-NC1*2:-NC1] = 1
        self.hedgecolor['color'] = color
        self.hedgecolor['level'] = colorlevel

    def refine_triangle_rg(self, isMarkedCell=None, options={}):
        NC = self.number_of_all_cells()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()

        color = self.hedgecolor
        node = self.entity('node')
        halfedge = self.ds.halfedge
        cstart = self.ds.cellstart
        subdomain = self.ds.subdomain
        hedge = self.ds.hedge
        hcell = self.ds.hcell

        if isMarkedCell is None:
            isMarkedCell = np.ones(NC, dtype=np.bool_)

        if self.hedgecolor is None:
            color = np.zeros(NE*2, dtype=np.int_)
            self.hedgecolor = color

        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']
        isMarkedCell[:cstart] = False

        isBlueHEdge = color == 3
        isYellowHedge = color == 2

        #得到加密半边并加密
        isMarkedHEdge0 = self.mark_halfedge(isMarkedCell, method='rg')
        isMarkedHEdge = isMarkedHEdge0 & ((color==0)|(color==1))

        NN1 = self.refine_halfedge(isMarkedHEdge)
        isMainHEdge = self.ds.main_halfedge_flag()
        NNE = NE+NN1

        color = np.r_[color, np.zeros(NN1*2, dtype = np.int_)]
        #标记的蓝色单元变成红色单元
        flag = isBlueHEdge & isMarkedHEdge0
        NE1 = flag.sum()

        color0 = self.refine_cell(flag, isMarkedHEdge,
                method='tri_coordinateCell', options=options)

        #修改半边颜色
        color = np.r_[color, np.zeros(NE1*4, dtype=np.int_)]
        color[color0] = 0

        #标记要生成新单元的单元
        NV = self.ds.number_of_vertices_of_all_cells()
        isBlueCell = NV == 4
        isNewCell = (NV == 4)|(NV == 6)
        isNewCell[:cstart] = False

        NC+=NE1*2
        NNE+=NE1*2
        isMarkedHEdge = np.r_[isMarkedHEdge, np.zeros(NN1*2+NE1*4, dtype=np.bool_)]

        #修改半边颜色(1)
        flag = isMarkedHEdge & isBlueCell[halfedge[:, 1]]
        tmp = np.where(flag)[0]
        #生成新单元
        flag = isMarkedHEdge & isNewCell[halfedge[:, 1]]#既是标记边又对应标记单元
        NC1 = flag.sum()
        self.refine_cell(isNewCell, flag, method='tri', options=options)

        #修改半边颜色
        color = np.r_[color, np.zeros(NC1*2, dtype = np.int_)]
        color[tmp] = 1
        color[halfedge[tmp, 3]]=3
        color[halfedge[color==3, 4]] = 2
        color[halfedge[color==2, 3]] = 1
        self.hedgecolor = color

    def coarsen_triangle_rg(self, isMarkedCell, options={}):

        NC = self.number_of_all_cells()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']

        color = self.hedgecolor
        halfedge = self.entity('halfedge')
        cstart = self.ds.cellstart
        isMarkedCell[:cstart] = True

        #标记红色单元要删除的半边
        flag = (clevel[halfedge[:, 1]]==clevel[halfedge[halfedge[:, 4], 1]])
        flag = (hlevel[:]==0) & (clevel[halfedge[:, 1]]
                >0) & flag & isMarkedCell[halfedge[:, 1]]
        flag = flag & flag[halfedge[:, 4]]#自身和对边都是标记单元的半边

        isRCell = np.ones(NC, dtype=np.bool_)
        np.logical_and.at(isRCell, halfedge[:, 1], flag)#周围都被标记的中心单元

        isMarkedHEdge = isRCell[halfedge[:, 1]]
        isMarkedHEdge[halfedge[isMarkedHEdge, 4]] = True

        #标记绿色单元要删除的半边
        flag = ((color==2) | (color==3)) & isMarkedCell[halfedge[:, 1]]
        flag = flag & flag[halfedge[:, 4]]
        isMarkedHEdge[flag] = True

        #删除单元
        isMarkedHEdge = self.coarsen_cell(isMarkedCell, isMarkedHEdge, method='tri')

        color = color[~isMarkedHEdge]#更新颜色
        color[color==1] = 0
        color[halfedge[color==3, 2]] = 1
        color[halfedge[color==2, 3]] = 1

        #进一步标记需要删除的边
        NC = self.number_of_all_cells()
        flag = (halfedge[halfedge[halfedge[halfedge[:, 2], 4], 2],
            4]==np.arange(len(halfedge)))
        flag = flag & (hlevel[:]==0) & (hlevel[halfedge[:, 4]]!=0)

        while True:
            count = np.zeros(NC, dtype=np.int_)
            np.add.at(count, halfedge[:, 1], flag)

            NV = self.ds.number_of_vertices_of_all_cells()
            flag0 = (count==1) & (NV==6)
            flag[flag0[halfedge[:, 1]]] = False
            flag[flag0[halfedge[halfedge[:, 4], 1]]] = False
            if (~flag0).all():
                break

        #修改颜色
        color[halfedge[flag, 4]] = 0
        color = color[~flag]
        self.coarsen_halfedge(flag)

        #标记要生成新单元的单元
        NV = self.ds.number_of_vertices_of_all_cells()
        isBlueCell = NV==4
        isRedCell = NV==6
        isNewCell = (NV == 4)|(NV == 6)
        isNewCell[:cstart] = False
        isRedCell[:cstart] = False

        #生成新单元
        flag1 = (hlevel[:]>0) & (hlevel[:]!=hlevel[halfedge[:, 4]])
        flag1 = flag1 & (hlevel[halfedge[:, 3]]==0)
        flag1 = flag1 & (hlevel[halfedge[halfedge[:, 3], 4]]>0)
        flag = flag1 & isNewCell[halfedge[:, 1]]#既是标记边又对应标记单元
        tmp = flag1 & isBlueCell[halfedge[:, 1]]#既是标记边又对应标记单元
        tmp, = np.where(tmp)
        NC1 = flag.sum()

        flag = (hlevel[:]==0) & (hlevel[halfedge[:, 2]]>0) & (hlevel[halfedge[:,
            4]]>0) & isBlueCell[halfedge[:, 1]]
        flag = flag | ((hlevel[:]==0) & isRedCell[halfedge[:, 1]])
        flag = flag[halfedge[:, 3]]

        self.refine_cell(isNewCell, flag, method='tri', options=options)

        #修改半边颜色
        color = np.r_[color, np.zeros(NC1*2, dtype = np.int_)]
        color[tmp] = 1
        color[halfedge[tmp, 3]]=3
        color[halfedge[color==3, 4]] = 2
        color[halfedge[color==2, 3]] = 1
        self.hedgecolor = color

    def refine_triangle_nvb(self, isMarkedCell=None, options={}):
        NC = self.number_of_all_cells()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()

        color = self.hedgecolor
        node = self.entity('node')
        halfedge = self.ds.halfedge

        if isMarkedCell is None:
            isMarkedCell = np.ones(NC, dtype=np.bool_)

        if color is None:
            color = np.zeros(NE*2, dtype=np.int_)
            nex = halfedge[:, 2]
            pre = halfedge[:, 3]
            l = node[halfedge[:, 0]]-node[halfedge[pre, 0]]
            l = np.linalg.norm(l, axis=1)
            color[(l>l[nex]) & (l>l[pre])] = 1
            self.hedgecolor = color

        cstart = self.ds.cellstart
        subdomain = self.ds.subdomain
        hedge = self.ds.hedge
        hcell = self.ds.hcell

        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']
        isMarkedCell[:cstart] = False

        if ('HB' in options) and (options['HB'] is not None):
            HB = np.zeros((NC-cstart, 2), dtype=np.int)
            HB[:, 0] = np.arange(NC-cstart)
            HB[:, 1] = np.arange(NC-cstart)
            options['HB'] = HB

        isMarkedHEdge = self.mark_halfedge(isMarkedCell, method='nvb')
        flag = np.array(True)
        while flag.any():
            #加密半边
            flag = isMarkedHEdge & ((color==1)|(halfedge[:, 1]<cstart))
            flag = flag & flag[halfedge[:, 4]]

            #加密半边
            NE1 = self.refine_halfedge(flag)*2

            #加密单元
            NV = self.ds.number_of_vertices_of_all_cells()
            isNewCell = NV==4

            flag = np.r_[flag, np.zeros(NE1, dtype=np.bool_)]
            flag = flag & (halfedge[:, 1]>=cstart)

            NE2 = flag.sum()
            isMarkedHEdge = np.r_[isMarkedHEdge, np.zeros(NE1+NE2*2, dtype=np.bool_)]

            #修改半边颜色
            color = np.r_[color, np.zeros(NE1, dtype=np.int)]
            color[flag]=0
            color[halfedge[flag, 2]] = 1
            color[halfedge[halfedge[flag, 3], 3]] = 1
            color = np.r_[color, np.zeros(NE2*2, dtype=np.int)]
            self.hedgecolor = color

            self.refine_cell(isNewCell, flag, method='tri', options=options)


    def coarsen_triangle_nvb(self, isMarkedCell, options={}):
        NC = self.number_of_all_cells()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        hcell = self.ds.hcell
        hedge = self.ds.hedge
        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']

        color = self.hedgecolor
        if color is None:
            return 
        node = self.entity('node')
        halfedge = self.entity('halfedge')
        cstart = self.ds.cellstart
        subdomain = self.ds.subdomain
        isMainHEdge = self.ds.main_halfedge_flag()
        isMarkedCell[:cstart] = True

        nex, pre, opp = halfedge[:, 2], halfedge[:, 3], halfedge[:, 4]

        #标记要删除的半边
        isMarkedHEdge = isMarkedCell[halfedge[:, 1]] & (hlevel[:]==0) & (color[:]==0)
        isMarkedHEdge = isMarkedHEdge & (clevel[halfedge[:, 1]]>0)
        isMarkedHEdge = isMarkedHEdge & isMarkedHEdge[halfedge[:, 4]]

        #只考虑对着的四个四边形都被标记的情况或者两个边界上的三角形都被标记的情况
        isMarkedHEdge = isMarkedHEdge & (color[halfedge[:, 3]] == 1)

        isMarkedHEdge = isMarkedHEdge & (isMarkedHEdge[opp[nex[opp[nex]]]] | (halfedge[opp[nex], 1] < cstart))
        isMarkedHEdge[opp[isMarkedHEdge]] = True

        if ('HB' in options) and (options['HB'] is not None):
            HB = np.zeros((NC-cstart, 2), dtype=np.int)
            HB[:, 0] = np.arange(NC-cstart)
            HB[:, 1] = np.arange(NC-cstart)
            options['HB'] = HB

        isMarkedHEdge = self.coarsen_cell(isMarkedCell, isMarkedHEdge, method='tri', options=options)

        #修改颜色
        color = color[~isMarkedHEdge]
        flag = (color==1) & (color==1)[halfedge[:, 2]]
        color[flag] = 0
        color[halfedge[flag, 2]] = 0
        color[halfedge[flag, 3]] = 1

        #进一步标记要移除的半边
        NV = self.ds.number_of_vertices_of_all_cells()
        nex = halfedge[:, 2]
        opp = halfedge[:, 4]
        flag = (opp[nex[opp[nex]]] == np.arange(len(halfedge))) & (NV[halfedge[:, 1]]==4)
        flag = flag & (color==1)[halfedge[:, 2]]
        flag[opp[nex[flag]]]=True

        #粗化半边
        self.coarsen_halfedge(flag)

        #修改颜色
        color = color[~flag]
        self.hedgecolor = color

    def adaptive_options(
            self,
            method='mean',
            maxrefine=3,
            maxcoarsen=0,
            theta=1.0,
            maxsize=1e-2,
            minsize=1e-12,
            data=None,
            HB=True,
            imatrix=False,
            disp=True
            ):

        options = {
                'method': method,
                'maxrefine': maxrefine,
                'maxcoarsen': maxcoarsen,
                'theta': theta,
                'maxsize': maxsize,
                'minsize': minsize,
                'data': data,
                'HB': HB,
                'imatrix': imatrix,
                'disp': disp
            }
        return options

    def adaptive(self, eta, options, method='poly'):

        if options['HB'] is True:
            HB = np.zeros((len(eta), 2), dtype=np.int)
            HB[:, 0] = np.arange(len(eta))
            HB[:, 1] = np.arange(len(eta))
            options['HB'] = HB

        NC = self.number_of_all_cells()
        options['numrefine'] = np.zeros(NC, dtype=np.int8)
        theta = options['theta']
        cellstart = self.ds.cellstart
        if options['method'] == 'mean':
            options['numrefine'][cellstart:] = np.around(
                    np.log2(eta/(theta*np.mean(eta)))/1
                )
            #options['numrefine'][cellstart:] = np.around(
            #        np.log2(eta/(theta*np.mean(eta)))
            #    )
        elif options['method'] == 'max':
            options['numrefine'][cellstart:] = np.around(
                    np.log2(eta/(theta*np.max(eta)))
                )
        elif options['method'] == 'median':
            options['numrefine'][cellstart:] = np.around(
                    np.log2(eta/(theta*np.median(eta)))
                )
        elif options['method'] == 'min':
            options['numrefine'][cellstart:] = np.around(
                    np.log2(eta/(theta*np.min(eta)))
                )
        elif options['method'] == 'numrefine':
            options['numrefine'][cellstart:] = eta
        elif isinstance(options['method'], float):
            val = options['method']
            options['numrefine'][cellstart:] = np.around(
                    np.log2(eta/val)
                )
        else:
            raise ValueError(
                    "I don't know anyting about method %s!".format(
                        options['method']))

        flag = options['numrefine'] > options['maxrefine']
        options['numrefine'][flag] = options['maxrefine']
        flag = options['numrefine'] < -options['maxcoarsen']
        options['numrefine'][flag] = -options['maxcoarsen']

        isMarkedCell = (options['numrefine'] > 0)

        while np.any(isMarkedCell):
            if method=='poly':
                self.refine_poly(isMarkedCell,options=options)
            if method=='quad':
                self.refine_quad(isMarkedCell,options=options)
            if method=='nvb':
                self.refine_triangle_nvb(isMarkedCell,options=options)
            if method=='rg':
                self.refine_triangle_rg(isMarkedCell,options=options)
            isMarkedCell = (options['numrefine'] > 0)

        # coarsen
        if options['maxcoarsen'] > 0:
            isMarkedCell = (options['numrefine'] < 0)
            while np.sum(isMarkedCell) > 0:
                NN0 = self.number_of_cells()
                if method=='poly':
                    self.coarsen_poly(isMarkedCell,options=options)
                if method=='quad':
                    self.coarsen_quad(isMarkedCell,options=options)
                if method=='nvb':
                    self.coarsen_triangle_nvb(isMarkedCell,options=options)
                if method=='rg':
                    self.coarsen_triangle_rg(isMarkedCell,options=options)
                NN = self.number_of_cells()
                if NN == NN0:
                    break
                isMarkedCell = (options['numrefine'] < 0)

    def adaptive_refine(self, isMarkedCell, method="nvb", options={}):
        cstart = self.ds.cellstart
        NC = self.number_of_all_cells()
        isMarkedCell0 = np.zeros(NC, dtype=np.bool_)
        isMarkedCell0[cstart:] = isMarkedCell
        if method=='nvb':
            self.refine_triangle_nvb(isMarkedCell0, options=options)
        elif method=='rg':
            self.refine_triangle_rg(isMarkedCell0, options=options)
        elif method=='quad':
            self.refine_quad(isMarkedCell0, options=options)
        elif method=='poly':
            self.refine_poly(isMarkedCell0, options=options)

    def adaptive_coarsen(self, isMarkedCell, method="nvb", options={}):
        cstart = self.ds.cellstart
        NC = self.number_of_all_cells()
        isMarkedCell0 = np.zeros(NC, dtype=np.bool_)
        isMarkedCell0[cstart:] = isMarkedCell
        if method=='nvb':
            self.coarsen_triangle_nvb(isMarkedCell0, options=options)
        elif method=='rg':
            self.coarsen_triangle_rg(isMarkedCell0, options=options)
        elif method=='quad':
            self.coarsen_quad(isMarkedCell0, options=options)
        elif method=='poly':
            self.coarsen_poly(isMarkedCell0, options=options)

    def adjust_number(self, isMarked, method='node'):
        L = len(isMarked)
        l = (~isMarked).sum()
        idxmap = np.zeros(L, dtype=np.int_)-10
        idxmap[~isMarked] = np.arange(l)
        halfedge = self.entity('halfedge')
        if method == 'node':
            halfedge[:, 0] = idxmap[halfedge[:, 0]]
        elif method == 'cell':
            halfedge[:, 1] = idxmap[halfedge[: ,1]]
        elif method == 'halfedge':
            halfedge[:, 2:] = idxmap[halfedge[:, 2:]]

    def uniform_refine(self, n=1):
        if self.ds.NV == 3:
            for i in range(n):
                self.refine_triangle_rg()
        else:
            for i in range(n):
                self.refine_poly()

    def tri_uniform_refine(self, n=1, method="rg"):
        if method == 'rg':
            for i in range(n):
                self.refine_triangle_rg()
        elif method == 'nvb':
            for i in range(n*2):
                self.refine_triangle_nvb()
        else:
            raise ValueError("refine type error! \"rg\" or \" nvb\"")


    def halfedge_direction(self):
        node = self.entity('node')
        halfedge = self.entity('halfedge')

        v = node[halfedge[:, 0]] - node[halfedge[halfedge[:, 4], 0]]
        return v

    def halfedge_length(self):
        v = self.halfedge_direction()
        l = np.linalg.norm(v, axis=1)
        return l

    def refine_marker(self, eta, theta, method="L2"):
        nc = self.number_of_all_cells()
        isMarkedCell = np.zeros(nc, dtype=np.bool_)
        isMarkedCell[self.ds.cellstart:] = mark(eta, theta, method=method)
        return isMarkedCell

    def mark_helper(self, idx):
        NC = self.number_of_cells()
        flag = np.zeros(NC, dtype=np.bool_)
        flag[idx] = True
        nc = self.number_of_all_cells()
        isMarkedCell = np.zeros(nc, dtype=np.bool_)
        isMarkedCell[self.ds.cellstart:] = flag
        return isMarkedCell

    def add_halfedge_plot(self, axes,
        index=None, showindex=False,
        nodecolor='r', edgecolor=['r', 'k'], markersize=20,
        fontsize=20, fontcolor='k', multiindex=None, linewidth=0.5):

        show_halfedge_mesh(axes, self,
                index=index, showindex=showindex,
                nodecolor=nodecolor, edgecolor=edgecolor, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor, 
                multiindex=multiindex, linewidth=linewidth)

    def split_to_trimesh(self):
        """
        @brief 在每个单元中心生成一个节点，将单元分成多个三角形单元.
                         3-------2
                         |\     /|
                         | \   / |
                         |  \ /  |
                         |   4   |
                         |  / \  |
                         | /   \ |
                         |/     \|
                         0-------1
                (0, 1, 2, 3) splite to [(4, 0, 1), (4, 1, 2), (4, 2, 3), (4, 3, 0)]
        @return 分割后的三角形网格，以及每个三角形单元的第一条边对应原来多边形
                网格中的半边编号
        """
        node = self.entity('node')
        halfedge = self.entity('halfedge')
        cstart = self.ds.cellstart
        NN = self.number_of_nodes()
        NC = self.number_of_cells()

        newnode = np.zeros((NC+NN, node.shape[1]), dtype=np.float_)
        newnode[:NN] = node[:]
        newnode[NN:] = self.entity_barycenter('cell')

        flag = halfedge[:, 1]>=cstart
        NTC = np.sum(flag)
        newcell = np.zeros((NTC, 3), dtype=np.int_)
        newcell[:, 2] = halfedge[flag, 0]
        newcell[:, 1] = halfedge[halfedge[flag, 4], 0]
        newcell[:, 0] = halfedge[flag, 1]-cstart+NN
        return TriangleMesh(newnode, newcell), np.where(flag)[0]

    def split_to_trimesh_1(self):
        """
        @brief 在每个单元中心生成一个节点，将单元分成多个三角形单元.
                         3-------2
                         |\     /|
                         | \   / |
                         |  \ /  |
                         |   4   |
                         |  / \  |
                         | /   \ |
                         |/     \|
                         0-------1
                (0, 1, 2, 3) split to [(4, 0, 1), (4, 1, 2), (4, 2, 3), (4, 3, 0)]
        @return 分割后的三角形网格，以及每个三角形单元的第一条边对应原来多边形
                网格中的半边编号
        """
        node     = self.entity('node')
        halfedge = self.entity('halfedge')
        cstart   = self.ds.cellstart
        hcell    = self.ds.hcell

        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        newnode      = np.zeros((NC+NN, node.shape[1]), dtype=np.float_)
        newnode[:NN] = node[:]
        newnode[NN:] = self.entity_barycenter('cell')

        NV = self.ds.number_of_vertices_of_cells()
        NTC = np.sum(NV)

        idx, _, num = self.ds.cell_to_halfedge(returnLocalnum=True)
        newcell       = np.zeros((NTC, 3), dtype=np.int_)
        newcell[:, 2] = halfedge[idx, 0]
        newcell[:, 1] = halfedge[halfedge[idx, 4], 0]
        newcell[:, 0] = halfedge[idx, 1]-cstart+NN
        return TriangleMesh(newnode, newcell), idx, num[idx]

    def to_dual_mesh(self, projection=None):
        isbdedge = self.ds.boundary_edge_flag()
        isbdnode = self.ds.boundary_node_flag()

        NE = self.number_of_edges()
        NN = self.number_of_nodes()
        cb = self.entity_barycenter('cell')
        eb = self.entity_barycenter('edge')[isbdedge]
        nb = self.entity('node')[isbdnode]
        if projection is not None:
            eb = projection(eb)
            nb = projection(nb)
        NN0 = len(cb)
        NN1 = len(eb)
        NN2 = len(nb)

        e2newnode = -np.ones(NE, dtype=np.int_)
        e2newnode[isbdedge] = np.arange(NN0, NN0+NN1)
        h2e = self.ds.halfedge_to_edge()
        h2newnode = e2newnode[h2e]

        n2newnode = -np.ones(NN, dtype=np.int_)
        n2newnode[isbdnode] = np.arange(NN0+NN1, NN0+NN1+NN2)

        node = DynamicArray(np.r_[cb, eb, nb], dtype=self.ftype)
        halfedge = self.entity('halfedge')
        cstart = self.ds.cellstart
        newHalfedge = DynamicArray(halfedge[:], dtype=np.int_)

        v = halfedge[:, 0]
        nex = halfedge[:, 2]
        opp = halfedge[:, 4]
        newHalfedge[:, 0] = halfedge[:, 1]-cstart
        newHalfedge[:, 1] = v[opp] + cstart
        newHalfedge[:, 2] = opp[halfedge[:, 3]] 
        newHalfedge[:, 3] = nex[opp]

        h0 = np.where(halfedge[:, 1] < cstart)[0]
        h1 = opp[h0]
        newHalfedge[h0, 0] = h2newnode[h0]

        newHalfedge0 = newHalfedge.increase_size(4*NN1)
        newHalfedge[h0, 2] = np.arange(NE*2, NE*2+NN1) 
        newHalfedge[h1, 3] = np.arange(NE*2+NN1, NE*2+NN1*2)
        
        newHalfedge0[:NN1, 0] = n2newnode[v[h1]]
        newHalfedge0[:NN1, 1] = newHalfedge[h0, 1]
        newHalfedge0[:NN1, 2] = newHalfedge[opp[halfedge[h0, 3]], 3]
        newHalfedge0[:NN1, 3] = h0
        newHalfedge0[:NN1, 4] = np.arange(NE*2+NN1*2, NE*2+NN1*3)

        newHalfedge0[NN1:NN1*2, 0] = h2newnode[h0]
        newHalfedge0[NN1:NN1*2, 1] = newHalfedge[h1, 1]
        newHalfedge0[NN1:NN1*2, 2] = h1
        newHalfedge0[NN1:NN1*2, 3] = newHalfedge[nex[h0], 2]
        newHalfedge0[NN1:NN1*2, 4] = np.arange(NE*2+NN1*3, NE*2+NN1*4)

        newHalfedge0[NN1*2:NN1*3, 0] = h2newnode[h0]
        newHalfedge0[NN1*2:NN1*3, 1] = halfedge[h0, 1]
        newHalfedge0[NN1*2:NN1*3, 2] = np.arange(NE*2+NN1*3, NE*2+NN1*4)
        newHalfedge0[NN1*2:NN1*3, 3] = newHalfedge[newHalfedge[NE*2:NE*2+NN1, 2], 4]
        newHalfedge0[NN1*2:NN1*3, 4] = np.arange(NE*2, NE*2+NN1)

        newHalfedge0[NN1*3:, 0] = n2newnode[v[h0]]
        newHalfedge0[NN1*3:, 1] = halfedge[h0, 1]
        newHalfedge0[NN1*3:, 2] = newHalfedge[newHalfedge[NE*2+NN1:NE*2+NN1*2, 3], 4]
        newHalfedge0[NN1*3:, 3] = np.arange(NE*2+NN1*2, NE*2+NN1*3)
        newHalfedge0[NN1*3:, 4] = np.arange(NE*2+NN1, NE*2+NN1*2)

        newhedge = self.ds.hedge.increase_size(2*NN1)
        newhedge[:] = np.arange(NE*2, NE*2+NN1*2)

        self.ds.subdomain = DynamicArray(np.r_[self.ds.subdomain[:cstart],
            np.ones(len(node))], dtype=np.int_)

        self.ds.halfedge = newHalfedge
        self.node = node

        self.ds.NE = len(newHalfedge)//2
        self.ds.NC = NN
        self.ds.NN = len(node)

        self.ds.hcell = DynamicArray((self.ds.NC+cstart, ), dtype=self.itype) 
        self.ds.hcell[self.ds.halfedge[:, 1]] = np.arange(2*self.ds.NE) 

        self.ds.hnode = DynamicArray((self.ds.NN, ), dtype=self.itype)
        self.ds.hnode[self.ds.halfedge[:, 0]] = np.arange(2*self.ds.NE)

    def to_vtk(self, fname=None):
        import vtk
        import vtk.util.numpy_support as vnp

        NC = self.number_of_cells()

        node = self.entity('node')
        if self.ds.NV is None:
            cell, cellLoc = self.ds.cell_to_node()
        else:
            cell = self.ds.cell_to_node()
            cellLoc = np.arange(NC+1)*3

        if node.shape[1] == 2:
            node = np.c_[node, np.zeros((len(node), 1), dtype=np.float_)]
        points = vtk.vtkPoints()
        points.SetData(vnp.numpy_to_vtk(node))

        uGrid = vtk.vtkUnstructuredGrid()
        uGrid.SetPoints(points)

        cells = np.split(cell, cellLoc[1:-1])
        vtk_cells = vtk.vtkCellArray()
        for c in cells:
            vtk_cell = vtk.vtkPolygon()
            vtk_cell.GetPointIds().SetNumberOfIds(len(c))
            for i, point_id in enumerate(c):
                vtk_cell.GetPointIds().SetId(i, point_id)
            vtk_cells.InsertNextCell(vtk_cell)
        uGrid.SetCells(vtk.VTK_POLYGON, vtk_cells)

        pdata = uGrid.GetPointData()
        if self.nodedata:
            nodedata = self.nodedata
            for key, val in nodedata.items():
                if val is not None:
                    if len(val.shape) == 2 and val.shape[1] == 2:
                        shape = (val.shape[0], 3)
                        val1 = np.zeros(shape, dtype=val.dtype)
                        val1[:, 0:2] = val
                    else:
                        val1 = val

                    if val1.dtype == np.bool_:
                        d = vnp.numpy_to_vtk(val1.astype(np.int_))
                    else:
                        d = vnp.numpy_to_vtk(val1[:])
                    d.SetName(key)
                    pdata.AddArray(d)

        if self.celldata:
            celldata = self.celldata
            cdata = uGrid.GetCellData()
            for key, val in celldata.items():
                if val is not None:
                    if len(val.shape) == 2 and val.shape[1] == 2:
                        shape = (val.shape[0], 3)
                        val1 = np.zeros(shape, dtype=val.dtype)
                        val1[:, 0:2] = val
                    else:
                        val1 = val

                    if val1.dtype == np.bool_:
                        d = vnp.numpy_to_vtk(val1.astype(np.int_))
                    else:
                        d = vnp.numpy_to_vtk(val1[:])

                    d.SetName(key)
                    cdata.AddArray(d)

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(uGrid)
        writer.Write()

    def grad_lambda(self):
        """

        Notes
        -----

        计算三角形网格上的重心坐标函数的梯度。
        """

        assert self.ds.NV == 3 # 必须是三角形网格

        node = self.entity('node')
        cell = self.entity('cell')
        NC = self.number_of_cells()

        v0 = node[cell[:, 2], :] - node[cell[:, 1], :]
        v1 = node[cell[:, 0], :] - node[cell[:, 2], :]
        v2 = node[cell[:, 1], :] - node[cell[:, 0], :]
        GD = self.geo_dimension()
        nv = np.cross(v2, -v1)
        Dlambda = np.zeros((NC, 3, GD), dtype=self.ftype)
        if GD == 2:
            length = nv
            W = np.array([[0, 1], [-1, 0]])
            Dlambda[:, 0, :] = v0@W/length.reshape((-1, 1))
            Dlambda[:, 1, :] = v1@W/length.reshape((-1, 1))
            Dlambda[:, 2, :] = v2@W/length.reshape((-1, 1))
        elif GD == 3:
            length = np.sqrt(np.square(nv).sum(axis=1))
            n = nv/length.reshape((-1, 1))
            Dlambda[:, 0, :] = np.cross(n, v0)/length.reshape((-1,1))
            Dlambda[:, 1, :] = np.cross(n, v1)/length.reshape((-1,1))
            Dlambda[:, 2, :] = np.cross(n, v2)/length.reshape((-1,1))
        return Dlambda

    ## @ingroup MeshGenerators
    @classmethod
    def from_interface_cut_box(cls, interface, box, nx=10, ny=10):
        """
        @brief 生成界面网格, 要求每个单元与界面只能交两个点或者不想交。
            步骤为:
                1. 生成笛卡尔网格
                2. 找到相交的半边
                3. 加密相交的半边
                4. 找到新生成的半边
                5. 对新生成的半边添加下一条半边或上一条半边
        @note : 1. 每个单元与界面只能交两个点或者不相交
                2. 相交的单元会被分为两个单元，界面内部的单元将继承原来单元的编号
        """

        from .QuadrangleMesh import QuadrangleMesh
        from .interface_mesh_generator import find_cut_point 

        ## 1. 生成笛卡尔网格
        N = (nx+1)*(ny+1)
        NC = nx*ny
        node = np.zeros((N,2))
        X, Y = np.mgrid[box[0]:box[1]:complex(0,nx+1), box[2]:box[3]:complex(0,ny+1)]
        node[:, 0] = X.flatten()
        node[:, 1] = Y.flatten()
        NN = len(node)

        idx = np.arange(N).reshape(nx+1, ny+1)
        cell = np.zeros((NC,4), dtype=np.int_)
        cell[:,0] = idx[0:-1, 0:-1].flat
        cell[:,1] = idx[1:, 0:-1].flat
        cell[:,2] = idx[1:, 1:].flat
        cell[:,3] = idx[0:-1, 1:].flat
        mesh = cls.from_mesh(QuadrangleMesh(node, cell))

        NN = mesh.number_of_nodes()
        NC = mesh.number_of_all_cells()
        NE = mesh.number_of_edges()

        node = mesh.node
        cell = mesh.entity('cell')
        halfedge = mesh.entity('halfedge')
        cstart = mesh.ds.cellstart

        isMainHEdge = mesh.ds.main_halfedge_flag()

        phiValue = interface(node[:])
        #phiValue[np.abs(phiValue) < 0.1*h**2] = 0.0
        phiSign = np.sign(phiValue)

        # 2. 找到相交的半边
        edge = mesh.entity('edge')
        isCutHEdge = phiValue[halfedge[:, 0]]*phiValue[halfedge[halfedge[:, 4], 0]] < 0 

        cutHEdge, = np.where(isCutHEdge&isMainHEdge)
        cutEdge = mesh.ds.halfedge_to_edge(cutHEdge)

        e0 = node[edge[cutEdge, 0]]
        e1 = node[edge[cutEdge, 1]]
        cutNode = find_cut_point(interface, e0, e1)

        mesh.refine_halfedge(isCutHEdge, newnode = cutNode)

        newHE = np.where((halfedge[:, 0] >= NN)&(halfedge[:, 1]>=cstart))[0]
        ## 注意这里要把 newHE 区分为界面内部和界面外部的
        cen = mesh.entity_barycenter('halfedge', index=newHE)
        isinnewHE = interface(cen)<0
        newHEin = newHE[isinnewHE]
        newHEout = newHE[~isinnewHE]

        idx = np.argsort(halfedge[newHEout, 1])
        newHE[::2] = newHEout[idx]
        idx = np.argsort(halfedge[newHEin, 1])
        newHE[1::2] = newHEin[idx]
        newHE = newHE.reshape(-1, 2)
        ################################################

        ne = len(newHE)
        NE = len(halfedge)//2

        mesh.number_cut_cell = ne

        halfedgeNew = halfedge.increase_size(ne*2)
        halfedgeNew[:ne, 0] = halfedge[newHE[:, 1], 0]
        halfedgeNew[:ne, 1] = halfedge[newHE[:, 1], 1]
        halfedgeNew[:ne, 2] = halfedge[newHE[:, 1], 2]
        halfedgeNew[:ne, 3] = newHE[:, 0] 
        halfedgeNew[:ne, 4] = np.arange(NE*2+ne, NE*2+ne*2)

        halfedgeNew[ne:, 0] = halfedge[newHE[:, 0], 0]
        halfedgeNew[ne:, 1] = np.arange(NC, NC+ne)
        halfedgeNew[ne:, 2] = halfedge[newHE[:, 0], 2]
        halfedgeNew[ne:, 3] = newHE[:, 1] 
        halfedgeNew[ne:, 4] = np.arange(NE*2, NE*2+ne) 

        halfedge[halfedge[newHE[:, 0], 2], 3] = np.arange(NE*2+ne, NE*2+ne*2)
        halfedge[halfedge[newHE[:, 1], 2], 3] = np.arange(NE*2, NE*2+ne)
        halfedge[newHE[:, 0], 2] = np.arange(NE*2, NE*2+ne)
        halfedge[newHE[:, 1], 2] = np.arange(NE*2+ne, NE*2+ne*2)

        isNotOK = np.ones(ne, dtype=np.bool_)
        current = np.arange(NE*2+ne, NE*2+ne*2)
        while np.any(isNotOK):
            halfedge[current[isNotOK], 1] = np.arange(NC, NC+ne)[isNotOK]
            current[isNotOK] = halfedge[current[isNotOK], 2]
            isNotOK = current != np.arange(NE*2+ne, NE*2+ne*2)


        #增加主半边
        mesh.ds.hedge.extend(np.arange(NE*2, NE*2+ne))

        #更新subdomain
        subdomainNew = mesh.ds.subdomain.increase_size(ne)
        subdomainNew[:] = mesh.ds.subdomain[halfedge[newHE[:, 0], 1]]

        #更新起始边
        mesh.ds.hcell.increase_size(ne)
        mesh.ds.hcell[halfedge[:, 1]] = np.arange(len(halfedge)) # 的编号

        mesh.ds.NN = mesh.node.size
        mesh.ds.NC += ne 
        mesh.ds.NE = halfedge.size//2

        mesh.init_level_info()
        return mesh

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

        print("edge:")
        edge = self.entity('edge')
        for i, val in enumerate(edge):
            print(i, ":", val)

        print("edge2cell:")
        edge2cell = self.ds.edge_to_cell()
        for i, val in enumerate(edge2cell):
            print(i, ":", val)

        cell, cellLocation = self.entity('cell')
        print('cell:', cell)
        print('cellLocation:', cellLocation)
        cell2edge, cellLocation = self.ds.cell_to_edge()
        print("cell2edge:", cell2edge)

HalfEdgeMesh2d.set_ploter('polygon2d')

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

        self.hcell = DynamicArray((NC, ), dtype=self.itype) # hcell[i] is the index of one face of i-th cell
        self.hcell[halfedge[:, 1]] = np.arange(2*self.NE) # 的编号

        self.hnode = DynamicArray((NN, ), dtype=self.itype)
        self.hnode[halfedge[:, 0]] = np.arange(2*self.NE)

        flag = halfedge[:, 4] - np.arange(2*self.NE) > 0
        self.hedge = DynamicArray(np.arange(self.NE*2)[flag])
        flag = subdomain[halfedge[self.hedge, 1]] < 1
        self.hedge[flag] = halfedge[self.hedge[flag], 4]

        self.NV = NV

    def number_of_all_cells(self):
        return len(self.subdomain)
    
    def number_of_cells(self):
        return self.number_of_all_cells() - self.cellstart 

    def number_of_edges(self):
        return len(self.halfedge)//2
    
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
            val = np.ones(hflag.sum(), dtype=np.bool_)
            I = halfedge[hflag, 1] - cstart
            J = halfedge[hflag, 0]
            cell2node = csr_matrix((val, (I, J)), shape=(NC, NN))
            return cell2node
        elif self.NV is None: # polygon mesh
            NV = self.number_of_vertices_of_cells()
            cellLocation = np.zeros(NC+1, dtype=self.itype)
            cellLocation[1:] = np.cumsum(NV)
            cell2node = np.zeros(cellLocation[-1], dtype=self.itype)
            current = self.hcell[cstart:].copy()
            idx = cellLocation[:-1].copy()
            cell2node[idx] = halfedge[halfedge[current, 3], 0]
            NV0 = np.ones(NC, dtype=self.itype)
            isNotOK = NV0 < NV
            while np.any(isNotOK):
               idx[isNotOK] += 1
               NV0[isNotOK] += 1
               cell2node[idx[isNotOK]] = halfedge[current[isNotOK], 0]
               current[isNotOK] = halfedge[current[isNotOK], 2]
               isNotOK = (NV0 < NV)
            return cell2node, cellLocation
        elif self.NV == 3: # tri mesh
            cell2node = np.zeros([NC, 3], dtype = np.int_)
            current = halfedge[self.hcell[cstart:], 2]
            cell2node[:, 0] = halfedge[current, 0]
            current = halfedge[current, 2]
            cell2node[:, 1] = halfedge[current, 0]
            current = halfedge[current, 2]
            cell2node[:, 2] = halfedge[current, 0]
            return cell2node
        elif self.NV == 4: # quad mesh
            cell2node = np.zeros([NC, 4], dtype = np.int_)
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
        subdomain = self.subdomain
        hflag = subdomain[halfedge[:, 1]] > 0
        hedge = self.hedge

        J = np.zeros(2*NE, dtype=self.itype)
        J[hedge] = range(NE)
        J[halfedge[hedge, 4]] = range(NE)
        if return_sparse:
            val = np.ones(2*NE, dtype=np.bool_)
            I = halfedge[hflag, 1] - cstart
            cell2edge = csr_matrix((val[hflag], (I,
                J[hflag])), shape=(NC, NE), dtype=np.bool_)
            return cell2edge
        elif self.NV is None:
            NV = self.number_of_vertices_of_cells()
            cellLocation = np.zeros(NC+1, dtype=self.itype)
            cellLocation[1:] = np.cumsum(NV)

            cell2edge = np.zeros(cellLocation[-1], dtype=self.itype)
            current = self.hcell[cstart:].copy()
            idx = cellLocation[:-1].copy()
            cell2edge[idx] = J[current]
            NV0 = np.ones(NC, dtype=self.itype)
            isNotOK = NV0 < NV
            while np.any(isNotOK):
                current[isNotOK] = halfedge[current[isNotOK], 2]
                idx[isNotOK] += 1
                NV0[isNotOK] += 1
                cell2edge[idx[isNotOK]] = J[current[isNotOK]]
                isNotOK = (NV0 < NV)
            return cell2edge, cellLocation
        elif self.NV == 3: # tri mesh
            cell2edge = np.zeros([NC, 3], dtype = np.int_)
            current = self.hcell[cstart:]
            cell2edge[:, 0] = J[current]
            cell2edge[:, 1] = J[halfedge[current, 2]]
            cell2edge[:, 2] = J[halfedge[current, 3]]
            return cell2edge
        elif self.NV == 4: # quad mesh
            cell2edge = np.zeros([NC, 4], dtype=np.int_)
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

    def cell_to_edge_sign(self, return_sparse=False):
        NE = self.NE
        NC = self.NC
        NEC = self.number_of_edges_of_cells()

        edge2cell = self.edge_to_cell()
        if return_sparse == False:
            if self.NV is None:
                cellLocation = np.zeros(NC+1, dtype=self.itype)
                cellLocation[1:] = np.cumsum(NEC)
                cell2edgeSign = np.zeros(cellLocation[-1], dtype=np.bool_)
                cell2edgeSign[cellLocation[edge2cell[:, 0]] + edge2cell[:, 2]]=True
            else:
                cell2edgeSign = np.zeros((NC, NEC), dtype=np.bool_)
                cell2edgeSign[edge2cell[:, 0], edge2cell[:, 2]] = True
        else:
            val = np.ones(NE, dtype=np.bool_)
            cell2edgeSign = csr_matrix(
                    (val, (edge2cell[:, 0], range(NE))),
                    shape=(NC, NE), dtype=np.bool_)
        return cell2edgeSign

    def cell_to_face(self, return_sparse=True):
        return self.cell_to_edge(return_sparse=return_sparse)

    def cell_to_cell(self, return_sparse=True):
        NC = self.NC
        halfedge = self.halfedge
        cstart = self.cellstart
        subdomain = self.subdomain
        hflag = subdomain[halfedge[:, 1]] > 0
        hedge = self.hedge

        if return_sparse:
            flag = hflag & hflag[halfedge[:, 4]]
            val = np.ones(flag.sum(), dtype=np.bool_)
            I = halfedge[flag, 1] - cstart
            J = halfedge[halfedge[flag, 4], 1]-cstart
            cell2cell = coo_matrix((val, (I, J)), shape=(NC, NC))
            cell2cell+= coo_matrix((val, (J, I)), shape=(NC, NC))
            return cell2cell.tocsr()
        elif self.NV is None:
            NV = self.number_of_vertices_of_cells()
            cellLocation = np.zeros(NC+1, dtype=self.itype)
            cellLocation[1:] = np.cumsum(NV)
            cell2cell = np.zeros(cellLocation[-1], dtype=self.itype)
            current = self.hcell[cstart:].copy()
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
            return cell2cell
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
            current = self.hcell[cstart:].copy()
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
        val = np.ones(2*NE, dtype=np.bool_)
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

        if return_sparse:
            val = np.ones(hflag.sum(), dtype=np.bool_)
            I = halfedge[hflag, 0]
            J = halfedge[hflag, 1] - cstart
            node2cell = csr_matrix((val, (I.flat, J.flat)), shape=(NN, NC), dtype=np.bool_)
            return node2cell

    def cell_to_halfedge(self, returnLocalnum=False):
        """!
        @brief 半边在所属单元中的编号
        """
        cstart = self.cellstart
        halfedge = self.halfedge

        Location = self.number_of_vertices_of_cells()
        Location = np.r_[0, np.cumsum(Location)]
        c2he = np.zeros(Location[-1], dtype=np.int_)

        NC = self.NC 
        N = len(self.halfedge)
        halfedge2cellnum = np.zeros(N, dtype=np.int_) # 每条半边所在单元的编号
        hcell = self.hcell[cstart:].copy()
        isNotOK = np.ones(NC, dtype=np.bool_)
        i = 0
        while np.any(isNotOK):
            c2he[Location[:-1][isNotOK]+i] = hcell[isNotOK]
            halfedge2cellnum[hcell[isNotOK]] = i
            hcell[isNotOK] = halfedge[hcell[isNotOK], 2]
            isNotOK[isNotOK] = self.hcell[cstart:][isNotOK]!=hcell[isNotOK]
            i += 1
        if returnLocalnum:
            return c2he, Location, halfedge2cellnum
        else:
            return c2he, Location

    def halfedge_to_node_location_number(self):
        """!
        @brief 半边在所指向的顶点中的编号
        """
        N = len(self.halfedge)
        halfedge = self.halfedge
        halfedge2nodenum = np.zeros(N, dtype=np.int_) # 每条半边所在单元的编号
        hnode = self.hnode.copy()
        NN = len(hnode)
        isNotOK = np.ones(NC, dtype=np.bool_)
        i = 0
        while np.any(isNotOK):
            halfedge2nodenum[hnode[isNotOK]] = i
            hnode[isNotOK] = halfedge[hnode[isNotOK], 2]
            isNotOK[isNotOK] = self.hnode[isNotOK]!=hnode[isNotOK]
            i += 1
        return halfedge2nodenum

    def halfedge_to_cell_location_number(self):
        """!
        @brief 半边在所在单元中的编号
        """
        N = len(self.halfedge)
        halfedge = self.halfedge
        halfedge2cellnum = np.zeros(N, dtype=np.int_) # 每条半边所在单元的编号
        hcell = self.hcell.copy()
        NC = len(hcell) 
        isNotOK = np.ones(NC, dtype=np.bool_)
        i = 0
        while np.any(isNotOK):
            halfedge2cellnum[hcell[isNotOK]] = i
            hcell[isNotOK] = halfedge[hcell[isNotOK], 2]
            isNotOK[isNotOK] = self.hcell[isNotOK]!=hcell[isNotOK]
            i += 1
        return halfedge2cellnum

    def halfedge_to_edge(self, index = np.s_[:]):
        halfedge = self.halfedge
        hedge = self.hedge
        NE = halfedge.shape[0]//2

        halfedge2edge = np.zeros(len(halfedge), dtype=np.int_)
        halfedge2edge[hedge] = np.arange(NE)
        halfedge2edge[halfedge[hedge, 4]] = np.arange(NE)
        return halfedge2edge[index] 

    def boundary_node_flag(self):
        NN = self.NN
        halfedge =  self.halfedge # DynamicArray
        subdomain = self.subdomain # DynamicArray
        hflag = subdomain[halfedge[:, 1]] > 0
        isBdHEdge = hflag & (~hflag[halfedge[:, 4]])
        isBdNode = np.zeros(NN, dtype=np.bool_)
        isBdNode[halfedge[isBdHEdge, 0]] = True 
        return isBdNode

    def boundary_edge_flag(self):
        halfedge =  self.halfedge
        hedge = self.hedge
        subdomain = self.subdomain

        hflag = subdomain[halfedge[:, 1]] > 0
        isBdEdge = hflag[hedge] & (~hflag[halfedge[hedge, 4]])
        return isBdEdge 

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

        isBdCell = np.zeros(NC, dtype=np.bool_)
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
        isMainHEdge = np.zeros(2*self.NE, dtype=np.bool_)
        isMainHEdge[self.hedge[:]] = True
        return isMainHEdge

    def boundary_edge(self):
        edge = self.edge_to_node()
        return edge[self.boundary_edge_index()]
