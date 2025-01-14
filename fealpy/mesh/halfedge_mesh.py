from typing import Union, Optional
from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S
from .plot import Plotable
from .mesh_base import Mesh
from ..common import DynamicArray
from scipy.sparse import csr_matrix, coo_matrix
from scipy.spatial import cKDTree

class HalfEdgeMesh2d(Mesh, Plotable):
    def __init__(self, node, halfedge, NC=None, NV=None, nodedof=None,
                 initlevel=True):
        """

        Parameters
        ----------
        node : (NN, GD)
        halfedge : (2*NE, 5), 
            halfedge[i, 0]: the index of the vertex the i-th halfedge point to
            halfedge[i, 1]: the index of the cell the i-th halfedge blong to
            halfedge[i, 2]: the index of the next halfedge of i-th haledge 
            halfedge[i, 3]: the index of the prev halfedge of i-th haledge 
            halfedge[i, 4]: the index of the opposit halfedge of the i-th halfedge
        Notes
        -----
        这是一个用半边数据结构存储网格拓扑关系的类。半边数据结构表示的网格更适和
        网格的自适应算法的实现。

        这个类的核心数组都是动态数组， 可以根据网格实体数目的变化动态增加长度，
        理论上可有效减少内存开辟的次数。

        边界半边的对边是自身

        Reference
        ---------
        [1] https://github.com/maciejkula/dynarray/blob/master/dynarray/dynamic_array.py
        """

        super().__init__(TD=2, itype=halfedge.dtype, ftype=node.dtype)
        self.meshtype = 'tri'  # 为什么是三角形网格？
        self.type= 'TRI'

        #self.node = DynamicArray(node, dtype=self.ftype)
        self.node = node
        #self.halfedge = DynamicArray(halfedge, dtype=self.itype)
        self.halfedge = halfedge
        #self.hcell = DynamicArray((0, ), dtype=self.itype)
        #self.hnode = DynamicArray((0, ), dtype=self.itype)
        #self.hedge = DynamicArray((0, ), dtype=self.itype)

        #self.cell2ipt = None  # cell2ipoint
        #self.tree = None      # 网格节点构造树结构

        self.reinit(NN=node.shape[0], NC=NC, NV=NV)
        self.NV = NV if NV.shape[0]==1 else None if NV.shape[0]>1 else None 
        #self.NV = NV  # 这里是 None

        self.meshtype = 'halfedge2d'

        self.nodedata = {}
        self.edgedata = {}
        self.celldata = {}
        self.meshdata = {}
        self.halfedgedata = {}
        self.facedata = self.edgedata

        # 网格节点的自由度标记数组
        # 0: 固定点
        # 1: 边界上的点
        # 2: 区域内部的点
        if nodedof is not None:
            self.nodedata['dof'] = nodedof
        
        if initlevel:
            self.init_level_info()

        #if self.NV==3:
        #    self.shape_function = self._shape_function
        #    self.cell_shape_function = self._shape_function
        #    self.face_shape_function = self._shape_function
        #    self.edge_shape_function = self._shape_function
    @property
    def device(self):
        return bm.get_device(self.halfedge)

    def reinit(self, NN=None, NC=None, NV=None):
        """

        Note
        ----
        self.halfedge, 
        self.hcell: 每个单元对应的一个半边
        self.hedge: 每条边对应的一条半边有性质 hedge >= halfedge[hedge, 4]
        self.hnode: 到达每个节点的一个半边
        """
        self.cell2ipt = None   # cell2ipoint
        self.tree = None       # 网格节点构造树结构

        halfedge = self.halfedge

        self.NHE = len(halfedge)  # 半边个数
        self.NBE = bm.sum(halfedge[:, 4]==bm.arange(self.NHE))  # 边界边的个数
        self.NE = (self.NHE+self.NBE)//2

        self.NN = NN if NN is not None else bm.max(halfedge[:, 0])+1
        self.NC = NC if NC is not None else bm.max(halfedge[:, 1])+1

        #if len(self.hcell)<self.NC:
        #    self.hcell.increase_size(self.NC-len(self.hcell))
        #else:
        #    self.hcell.decrease_size(len(self.hcell)-self.NC)
        self.hcell = bm.arange(self.NC, dtype=self.itype)
        self.hcell[halfedge[:, 1]] = bm.arange(self.NHE, dtype=self.itype)

        #if len(self.hnode)<self.NN:
        #    self.hnode.increase_size(self.NN-len(self.hnode))
        #else:
        #    self.hnode.decrease_size(len(self.hnode)-self.NN)
        self.hnode = bm.arange(self.NN, dtype=self.itype)
        self.hnode[halfedge[:, 0]] = bm.arange(self.NHE, dtype=self.itype)

        #if len(self.hedge)<self.NE:
        #    self.hedge.increase_size(self.NE-len(self.hedge))
        #else:
        #    self.hedge.decrease_size(len(self.hedge)-self.NE)
        flag = bm.arange(self.NHE)-halfedge[:, 4] >= 0
        self.hedge = bm.arange(self.NHE, dtype=self.itype)[flag]

    def init_level_info(self):
        """
        @brief 初始化半边和单元的 level 
        """
        NN = self.number_of_nodes()
        NHE = self.number_of_halfedges()
        NC = self.number_of_cells() # 实际单元个数

        self.halfedgedata['level'] = bm.zeros(NHE, dtype=self.itype)
        self.celldata['level'] = bm.zeros(NC, dtype=self.itype)

        # 如果单元的角度大于 170 度， 设对应的半边层数为 1
        node = self.node
        halfedge = self.halfedge
        v0 = node[halfedge[halfedge[:, 2], 0]] - node[halfedge[:, 0]]
        v1 = node[halfedge[halfedge[:, 3], 0]] - node[halfedge[:, 0]]
        angle = bm.sum(v0*v1, axis=1)/bm.sqrt(bm.sum(v0**2, axis=1)*bm.sum(v1**2, axis=1))
        self.halfedgedata['level'][(angle < -0.98)] = 1 



    def top_dimension(self):
        return 2

    def number_of_edges(self):
        return self.NE 

    def number_of_cells(self):
        return self.NC 

    def number_of_halfedges(self):
        return self.NHE

    def number_of_boundary_nodes(self):
        return self.NBE

    def number_of_boundary_edges(self):
        return self.NBE

    def number_of_vertices_of_cells(self):
        NV = bm.zeros(self.NC, dtype=self.itype)
        bm.index_add(NV, self.halfedge[:, 1], 1)
        return NV

    def number_of_nodes_of_cells(self):
        return self.number_of_vertices_of_cells()

    def number_of_edges_of_cells(self):
        return self.number_of_vertices_of_cells()

    def number_of_faces_of_cells(self):
        return self.number_of_vertices_of_cells()

    def entity(self, etype=2, index=_S):
        if etype in {'cell', 2}:
            #return self.cell_to_node()[index]
            return self.halfedge
        elif etype in {'edge', 'face', 1}:
            return self.edge_to_node()[index]
        elif etype in {'halfedge'}:
            return self.halfedge # DynamicArray
        elif etype in {'node', 0}:
            return self.node # DynamicArrray
        else:
            raise ValueError("`entitytype` is wrong!")


    def convexity(self):
        """
        @brief 将网格中的非凸单元分解为凸单元
        """
        def angle(x, y):
            x = x/(bm.linalg.norm(x, axis=1)).reshape(len(x), 1)
            y = y/bm.linalg.norm(y, axis=1).reshape(len(y), 1)
            x1 = bm.concatenate((bm.zeros((x.shape[0],1)), x), axis=1)
            y1 = bm.concatenate([bm.zeros((y.shape[0],1)), y], axis=1)
            theta = bm.sign(bm.cross(x1, y1)[:,0])*bm.arccos((x*y).sum(axis=1))
            theta[theta<0]+=2*bm.pi
            theta[theta==0]+=bm.pi
            return theta

        halfedge = self.halfedge
        node = self.node

        hedge = self.hedge
        hcell = self.hcell

        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']
        while True:
            NC = self.number_of_cells()
            NHE = self.number_of_halfedges()

            end = halfedge[:, 0]
            start = halfedge[halfedge[:, 3], 0]
            vector = node[end] - node[start]#所有边的方向
            vectornex = vector[halfedge[:, 2]]#所有边的下一个边的方向

            angle0 = angle(vectornex, -vector)#所有边的反方向与下一个边的角度
            badHEdge, = bm.where(angle0 > bm.pi)#夹角大于170度的半边
            badCell, idx= bm.unique(halfedge[badHEdge, 1], return_index=True)#每个单元每次只处理中的一个
            badHEdge = badHEdge[idx]#现在坏半边的单元都不同
            badNode = halfedge[badHEdge, 0]
            NE1 = len(badHEdge)

            nex = halfedge[badHEdge, 2]
            pre = halfedge[badHEdge, 3]
            vectorBad = vector[badHEdge]#坏半边的方向
            vectorBadnex = vector[nex]#坏半边的下一个半边的方向

            anglenex = angle(vectorBadnex, -vectorBad)#坏边的夹角
            anglecur = anglenex/2#当前方向与角平分线的夹角
            angle_err_min = anglenex/2#与角平分线夹角的最小值
            goodHEdge = bm.zeros(NE1, dtype=self.itype)#最小夹角的边
            isNotOK = bm.ones(NE1, dtype = bm.bool)#每个单元的循环情况
            nex = halfedge[nex, 2]#从下下一个边开始
            while isNotOK.any():
                vectornex = node[halfedge[nex, 0]] - node[badNode]
                anglecur[isNotOK] = angle(vectorBadnex[isNotOK], vectornex[isNotOK])
                angle_err = abs(anglecur - anglenex/2)
                goodHEdge[angle_err<angle_err_min] = nex[angle_err<angle_err_min]#与角平分线夹角小于做小夹角的边做goodHEdge.
                angle_err_min[angle_err<angle_err_min] = angle_err[angle_err<angle_err_min]#更新最小角
                nex = halfedge[nex, 2]
                isNotOK[nex==pre] = False#循环到坏边的上上一个边结束
            #halfedgeNew = halfedge.increase_size(NE1*2)
            halfedgeNew = bm.zeros((NE1*2, 5), dtype=self.itype)
            halfedgeNew[:NE1, 0] = bm.copy(halfedge[goodHEdge, 0])
            halfedgeNew[:NE1, 1] = bm.copy(halfedge[badHEdge, 1])
            halfedgeNew[:NE1, 2] = bm.copy(halfedge[goodHEdge, 2])
            halfedgeNew[:NE1, 3] = bm.copy(badHEdge)
            halfedgeNew[:NE1, 4] = bm.arange(NHE+NE1, NHE+NE1*2)

            halfedgeNew[NE1:, 0] = bm.copy(halfedge[badHEdge, 0])
            halfedgeNew[NE1:, 1] = bm.arange(NC, NC+NE1)
            halfedgeNew[NE1:, 2] = bm.copy(halfedge[badHEdge, 2])
            halfedgeNew[NE1:, 3] = bm.copy(goodHEdge)
            halfedgeNew[NE1:, 4] = bm.arange(NHE, NHE+NE1, dtype=self.itype)
            self.halfedge = bm.concatenate([halfedge, halfedgeNew], axis=0)
            halfedge = self.halfedge

            halfedge[halfedge[goodHEdge, 2], 3] = bm.arange(NHE, NHE+NE1, dtype=self.itype)
            halfedge[halfedge[badHEdge, 2], 3] = bm.arange(NHE+NE1, NHE+NE1*2, dtype=self.itype)
            halfedge[badHEdge, 2] = bm.arange(NHE, NHE+NE1, dtype=self.itype)
            halfedge[goodHEdge, 2] = bm.arange(NHE+NE1, NHE+NE1*2, dtype=self.itype)
            isNotOK = bm.ones(NE1, dtype=bm.bool)
            nex = halfedge[len(halfedge)-NE1:, 2]
            while isNotOK.any():
                halfedge[nex[isNotOK], 1] = bm.arange(NC, NC+NE1, dtype=self.itype)[isNotOK]
                nex = halfedge[nex, 2]
                flag = (nex==bm.arange(NHE+NE1, NHE+NE1*2)) & isNotOK
                isNotOK[flag] = False

            #单元层
            #clevelNew = clevel.increase_size(NE1)
            clevelNew = bm.zeros(NE1, dtype=self.itype)
            clevelNew[:] = clevel[halfedge[badHEdge, 1]]
            self.celldata['level'] = bm.concatenate([clevel, clevelNew], axis=0)
            clevel = self.celldata['level']

            #半边层
            #hlevelNew = hlevel.increase_size(NE1*2)
            hlevelNew = bm.zeros(NE1*2, dtype=self.itype)
            hlevelNew[:NE1] = hlevel[goodHEdge]
            hlevelNew[NE1:] = hlevel[badHEdge]
            self.halfedgedata['level'] = bm.concatenate([hlevel, hlevelNew], axis=0)
            hlevel = self.halfedgedata['level']

            self.reinit()
            if len(badHEdge)==0:
                break

    def find_point(self, points, start=None): 
        raise NotImplementedError('The method find_point is not implemented!')

########################### 三角形网格的接口 ####################################
    def find_point_in_triangle_mesh(self, points, start=None):
        raise NotImplementedError('The method find_point_in_triangle_mesh is not implemented!')
 

    def interpolation_cell_data(self, mesh, datakey, itype="max"):
        raise NotImplementedError('The method interpolation_cell_data is not implemented!')

    def grad_shape_function(self, bc, p=1, index=_S, variables='x'):
        """
        @note 注意这里调用的实际上不是形状函数的梯度，而是网格空间基函数的梯度
        """
        if self.NV==3:
            R = bm.simplex_grad_shape_function(bc, p)
            if variables == 'x':
                Dlambda = self.grad_lambda(index=index)
                gphi = bm.einsum('...ij, kjm->...kim', R, Dlambda)
                return gphi #(NQ, NC, ldof, GD)
            elif variables == 'u':
                return R #(NQ, ldof, TD+1)
        else:
            pass

    def grad_lambda(self, index=_S):
        node = self.entity('node')
        cell = self.entity('cell')
        NC = self.number_of_cells() if index is _S else len(index)
        #NC = self.number_of_cells() if index == bm.s_[:] else len(index)
        import ipdb
        ipdb.set_trace()
        v0 = node[cell[index, 2]] - node[cell[index, 1]]
        v1 = node[cell[index, 0]] - node[cell[index, 2]]
        v2 = node[cell[index, 1]] - node[cell[index, 0]]
        GD = self.geo_dimension()
        nv = bm.linalg.cross(v1, v2)
        Dlambda = bm.zeros((NC, 3, GD), dtype=self.ftype)
        if GD == 2:
            length = nv
            W = bm.array([[0, 1], [-1, 0]])
            Dlambda[:, 0] = v0@W/length[:, None]
            Dlambda[:, 1] = v1@W/length[:, None]
            Dlambda[:, 2] = v2@W/length[:, None]
        elif GD == 3:
            length = bm.linalg.norm(nv, axis=-1, keepdims=True)
            n = nv/length
            Dlambda[:, 0] = bm.linalg.cross(n, v0)/length
            Dlambda[:, 1] = bm.cross(n, v1)/length
            Dlambda[:, 2] = bm.cross(n, v2)/length
        return Dlambda



    @classmethod
    def from_mesh(cls, mesh):
        """

        Notes
        -----
        输入一个其它类型数据结构的网格，转化为半边数据结构。如果 closed 为真，则
        表明输入网格是一个封闭的曲面网格；为假，则为开的网格，可以存在洞，或者无
        界的外界区域
        """
        mtype = mesh.meshtype
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        NV = mesh.number_of_vertices_of_cells()
        if mtype not in {'halfedge', 'halfedge2d'}:
            NE = mesh.number_of_edges()
            NBE = mesh.boundary_face_flag().sum()
            NHE = NE*2 - NBE

            node = mesh.node
            edge = mesh.edge
            cell = mesh.cell
            edge2cell = mesh.face_to_cell()

            isInEdge = edge2cell[:, 0] != edge2cell[:, 1]
            isBdEdge = ~isInEdge

            idx = bm.zeros((NHE, 2), dtype=mesh.itype)
            halfedge = bm.zeros((NHE, 5), dtype=mesh.itype)
            halfedge[:NHE-NBE, 0] = edge[isInEdge].flatten()
            halfedge[NHE-NBE:, 0] = edge[isBdEdge, 1]

            halfedge[0:NHE-NBE:2, 1] = edge2cell[isInEdge, 1]
            halfedge[1:NHE-NBE:2, 1] = edge2cell[isInEdge, 0]
            halfedge[NHE-NBE:, 1] = edge2cell[isBdEdge, 0]

            halfedge[0:NHE-NBE:2, 4] = bm.arange(1, NHE-NBE, 2)
            halfedge[1:NHE-NBE:2, 4] = bm.arange(0, NHE-NBE, 2)
            halfedge[NHE-NBE:, 4] = bm.arange(NHE-NBE, NHE)

            idx[0:NHE-NBE:2, 1] = edge2cell[isInEdge, 3]
            idx[1:NHE-NBE:2, 1] = edge2cell[isInEdge, 2]
            idx[NHE-NBE:, 1] = edge2cell[isBdEdge, 2]
            idx[:, 0] = halfedge[:,1]

            idx = bm.lexsort((idx[:,1], idx[:,0]))
            idx = bm.array(idx, dtype=mesh.itype)
            halfedge[idx, 2] = bm.roll(idx, -1)
            halfedge[idx, 3] = bm.roll(idx, 1)

            idx0 = bm.where(halfedge[halfedge[idx, 2], 1]!=halfedge[idx, 1])[0]
            idx1 = bm.where(halfedge[halfedge[idx, 3], 1]!=halfedge[idx, 1])[0]
            halfedge[idx[idx0], 2] = idx[idx1]
            halfedge[idx[idx1], 3] = idx[idx0]

            mesh = cls(node, halfedge, NC=NC, NV=NV)
            return mesh
        else:
            newMesh =  cls(mesh.node.copy(), mesh.halfedge.copy(), NC=NC, NV=mesh.ds.NV)
            newMesh.celldata['level'][:] = mesh.celldata['level']
            newMesh.halfedge['level'][:] = mesh.halfedgedata['level']
            return newMesh

    def main_halfedge_flag(self):
        isMainHEdge = bm.zeros(self.NHE, dtype=bm.bool)
        isMainHEdge[self.hedge[:]] = True
        return isMainHEdge


    def cell_to_node(self, return_sparse=False):
        NN = self.NN
        NC = self.NC
        halfedge = self.halfedge
        if return_sparse:
            NHE = self.number_of_halfedges()
            val = bm.ones(NHE, dtype=bm.bool)
            I = halfedge[:, 1]
            J = halfedge[:, 0]
            cell2node = csr_matrix((val, (I, J)), shape=(NC, NN))
            return cell2node
        elif self.NV is None: # polygon mesh
            NV = self.number_of_vertices_of_cells()
            cellLocation = bm.zeros(NC+1, dtype=self.itype)
            cellLocation[1:] = bm.cumsum(bm.array(NV), axis=0)
            cell2node = bm.zeros(cellLocation[-1], dtype=self.itype)
            current = bm.copy(self.hcell)
            idx = bm.copy(cellLocation[:-1])
            cell2node[idx] = halfedge[halfedge[current, 3], 0]
            NV0 = bm.ones(NC, dtype=self.itype)
            isNotOK = NV0 < NV
            while bm.any(isNotOK):
               idx[isNotOK] += 1
               NV0[isNotOK] += 1
               cell2node[idx[isNotOK]] = halfedge[current[isNotOK], 0]
               current[isNotOK] = halfedge[current[isNotOK], 2]
               isNotOK = (NV0 < NV)
            return bm.split(cell2node, cellLocation[1:-1])
        elif self.NV == 3: # tri mesh
            cell2node = bm.zeros([NC, 3], dtype = self.itype)
            current = halfedge[self.hcell[:], 2]
            cell2node[:, 0] = halfedge[current, 0]
            current = halfedge[current, 2]
            cell2node[:, 1] = halfedge[current, 0]
            current = halfedge[current, 2]
            cell2node[:, 2] = halfedge[current, 0]
            return cell2node
        elif self.NV == 4: # quad mesh
            cell2node = bm.zeros([NC, 4], dtype = self.itype)
            current = halfedge[self.hcell[:], 3]
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
        NHE = self.NHE
        NE = self.NE
        NC = self.NC

        halfedge = self.halfedge
        hedge = self.hedge

        J = bm.zeros(NHE, dtype=self.itype) # halfedge_to_edge
        J[hedge] = bm.arange(NE, dtype=self.itype)
        J[halfedge[hedge, 4]] = bm.arange(NE, dtype=self.itype)
        if return_sparse:
            val = bm.ones(NHE, dtype=bm.bool)
            I = halfedge[:, 1]
            cell2edge = csr_matrix((val, (I, J)), shape=(NC, NE))
            return cell2edge
        elif self.NV is None:
            NV = self.number_of_vertices_of_cells()
            cellLocation = bm.zeros(NC+1, dtype=self.itype)
            cellLocation[1:] = bm.cumsum(bm.array(NV), axis=0)

            cell2edge = bm.zeros(cellLocation[-1], dtype=self.itype)
            current = bm.copy(self.hcell)
            idx = bm.copy(cellLocation[:-1])
            cell2edge[idx] = J[current]
            NV0 = bm.ones(NC, dtype=self.itype)
            isNotOK = NV0 < NV
            while bm.any(isNotOK):
                current[isNotOK] = halfedge[current[isNotOK], 2]
                idx[isNotOK] += 1
                NV0[isNotOK] += 1
                cell2edge[idx[isNotOK]] = J[current[isNotOK]]
                isNotOK = (NV0 < NV)
            return cell2edge, cellLocation
        elif self.NV == 3: # tri mesh
            cell2edge = bm.zeros([NC, 3], dtype = self.itype)
            current = self.hcell
            cell2edge[:, 0] = J[current]
            cell2edge[:, 1] = J[halfedge[current, 2]]
            cell2edge[:, 2] = J[halfedge[current, 3]]
            return cell2edge
        elif self.NV == 4: # quad mesh
            cell2edge = bm.zeros([NC, 4], dtype=self.itype)
            current = self.hcell
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
        NHE = self.NHE
        hedge = self.hedge
        halfedge = self.halfedge

        if return_sparse:
            val = bm.ones(NHE, dtype=bm.bool)
            I = halfedge[:, 1]
            J = halfedge[halfedge[:, 4], 1]
            cell2cell = coo_matrix((val, (I, J)), shape=(NC, NC))
            cell2cell+= coo_matrix((val, (J, I)), shape=(NC, NC))
            return cell2cell.tocsr()
        elif self.NV is None:
            NV = self.number_of_vertices_of_cells()
            cellLocation = bm.zeros(NC+1, dtype=self.itype)
            cellLocation[1:] = bm.cumsum(bm.array(NV), axis=0)
            cell2cell = bm.zeros(cellLocation[-1], dtype=self.itype)
            current = bm.copy(self.hcell[:])
            idx = bm.copy(cellLocation[:-1])
            cell2cell[idx] = halfedge[halfedge[current, 4], 1]
            NV0 = bm.ones(NC, dtype=self.itype)
            isNotOK = NV0 < NV
            while isNotOK.sum() > 0:
                current[isNotOK] = halfedge[current[isNotOK], 2]
                idx[isNotOK] += 1
                NV0[isNotOK] += 1
                cell2cell[idx[isNotOK]] = halfedge[halfedge[current[isNotOK], 4], 1]
                isNotOK = (NV0 < NV)
            idx = bm.repeat(bm.arange(NC,dtype=self.itype), NV)
            flag = (cell2cell < 0)
            cell2cell[flag] = idx[flag]
            return cell2cell, cellLocation
        elif self.NV == 3: # tri mesh
            cell2cell = bm.zeros((NC, 3), dtype=self.itype)
            current = self.hcell[:]
            cell2cell[:, 0] = halfedge[halfedge[current, 4], 1]
            cell2cell[:, 1] = halfedge[halfedge[halfedge[current, 2], 4], 1]
            cell2cell[:, 2] = halfedge[halfedge[halfedge[current, 3], 4], 1]
            idx = bm.repeat(bm.arange(NC,dtype=self.itype), 3).reshape(NC, 3)
            flag = (cell2cell < 0)
            cell2cell[flag] = idx[flag]
            return cell2cell
        elif self.NV == 4: # quad mesh
            cell2cell = bm.zeros((NC, 4), dtype=self.itype)
            current = self.hcell[:]
            cell2cell[:, 3] = halfedge[halfedge[current, 4], 1]
            current = halfedge[current, 2]
            cell2cell[:, 0] = halfedge[halfedge[current, 4], 1]
            current = halfedge[current, 2]
            cell2cell[:, 1] = halfedge[halfedge[current, 4], 1]
            current = halfedge[current, 2]
            cell2cell[:, 2] = halfedge[halfedge[current, 4], 1]
            idx = bm.repeat(bm.arange(NC,dtype=self.itype), 4).reshape(NC, 4)
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
            edge = bm.zeros((NE, 2), dtype=self.itype)
            edge[:, 0] = halfedge[halfedge[hedge, 3], 0]
            edge[:, 1] = halfedge[hedge, 0]
            return edge
        else:
            val = bm.ones(NE, dtype=bm.bool)
            edge2node = coo_matrix((val, (bm.arange(NE, dtype=self.itype), halfedge[hedge, 0])), shape=(NE, NN))
            edge2node+= coo_matrix( (val, (bm.arange(NE, dtype=self.itype), halfedge[halfedge[hedge, 3], 0])), shape=(NE, NN))
            return edge2node.tocsr()

    def edge_to_edge(self):
        edge2node = self.edge_to_node(return_sparse=True)
        return edge2node@edge2node.transpose()

    def edge_to_cell(self):
        """
        @brief 计算每条边对应的两个单元和边在单元中的局部编号
        """
        NE = self.NE
        NC = self.NC
        NHE = self.NHE

        halfedge = self.halfedge
        hedge = self.hedge

        J = bm.zeros(NHE, dtype=self.itype) # halfedge to edge
        J[hedge] = bm.arange(NE, dtype=self.itype)
        J[halfedge[hedge, 4]] = bm.arange(NE, dtype=self.itype)

        edge2cell = bm.full((NE, 4), -1, dtype=self.itype)
        edge2cell[J[hedge], 0] = halfedge[hedge, 1]
        edge2cell[J[halfedge[hedge, 4]], 1] = halfedge[halfedge[hedge, 4], 1]

        isMainHEdge = self.main_halfedge_flag() 
        if self.NV is None:
            current = bm.copy(self.hcell[:])
            end = self.hcell[:] 
            lidx = bm.zeros_like(current)
            isNotOK = bm.ones_like(current, dtype=bm.bool)
            while bm.any(isNotOK):
                idx = J[current[isNotOK]]
                flag = isMainHEdge[current[isNotOK]]
                edge2cell[idx[flag], 2]  = lidx[isNotOK][flag]#[:, None]
                edge2cell[idx[~flag], 3] = lidx[isNotOK][~flag]
                current[isNotOK] = halfedge[current[isNotOK], 2]
                lidx[isNotOK] += 1
                isNotOK = (current != end)
            flag = edge2cell[:, -1]==-1
            edge2cell[flag, -1] = edge2cell[flag, -2]

        elif self.NV == 3:
            current = self.hcell[:]
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
            current = self.hcell[:]
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

        flag = edge2cell[:, -1] < 0 
        edge2cell[flag, 1] = edge2cell[flag, 0]
        edge2cell[flag, 3] = edge2cell[flag, 2]
        return edge2cell

    def node_to_node(self, return_sparse=True):
        NN = self.NN
        NE = self.NE
        NHE = self.NHE
        halfedge = self.halfedge
        I = halfedge[:, 0] 
        J = halfedge[halfedge[:, 3], 0] 
        val = bm.ones(NHE, dtype=bm.bool)
        node2node = csr_matrix((val, (I, J)), shape=(NN, NN))
        J = halfedge[halfedge[:, 2], 0]
        node2node += csr_matrix((val, (I, J)), shape=(NN, NN))
        node2node += csr_matrix((val, (I, I)), shape=(NN, NN))
        return node2node

    def node_to_edge(self, return_sparse=True):
        pass

    def node_to_cell(self, return_sparse=True):
        NN = self.NN
        NC = self.NC
        NHE = self.NHE
        halfedge =  self.halfedge

        if return_sparse:
            val = bm.ones(NHE, dtype=bm.bool)
            I = halfedge[:, 0]
            J = halfedge[:, 1]
            node2cell = csr_matrix((val, (I.flatten(), J.flatten())), shape=(NN, NC))
            return node2cell

    def cell_to_halfedge(self, returnLocalnum=False):
        """!
        @brief 每个单元的半边编号
        """
        halfedge = self.halfedge

        Location = self.number_of_vertices_of_cells()
        Location = bm.concatenate((bm.array([0]), bm.cumsum(Location, axis=0)))
        c2he = bm.zeros(Location[-1], dtype=self.itype)

        NC = self.NC 
        NHE = self.NHE 
        halfedge2cellnum = bm.zeros(NHE, dtype=self.itype) # 每条半边所在单元的编号
        hcell = bm.copy(self.hcell[:])
        isNotOK = bm.ones(NC, dtype=bm.bool)
        i = 0
        while bm.any(isNotOK):
            c2he[Location[:-1][isNotOK]+i] = hcell[isNotOK]
            halfedge2cellnum[hcell[isNotOK]] = i
            hcell[isNotOK] = halfedge[hcell[isNotOK], 2]
            isNotOK1 = bm.copy(isNotOK) 
            isNotOK[isNotOK1] = self.hcell[:][isNotOK1]!=hcell[isNotOK1]
            i += 1
        if returnLocalnum:
            return c2he, Location, halfedge2cellnum
        else:
            return c2he, Location

    #def halfedge_to_node_location_number(self):
    #    """!
    #    @brief 半边在所指向的顶点中的编号
    #    """
    #    N = len(self.halfedge)
    #    NC = self.NC
    #    halfedge = self.halfedge
    #    halfedge2nodenum = bm.zeros(N, dtype=self.itype) # 每条半边所在单元的编号
    #    hnode = bm.copy(self.hnode)
    #    NN = len(hnode)
    #    isNotOK = bm.ones(NN, dtype=bm.bool)
    #    i = 0
    #    while bm.any(isNotOK):
    #        halfedge2nodenum[hnode[isNotOK]] = i
    #        hnode[isNotOK] = halfedge[hnode[isNotOK], 2]
    #        isNotOK1 = bm.copy(isNotOK)
    #        isNotOK[isNotOK1] = self.hnode[isNotOK1]!=hnode[isNotOK1]
    #        i += 1
    #    return halfedge2nodenum

    def halfedge_to_cell_location_number(self):
        """!
        @brief 半边在所在单元中的编号
        """
        N = len(self.halfedge)
        halfedge = self.halfedge
        halfedge2cellnum = bm.zeros(N, dtype=self.itype) # 每条半边所在单元的编号
        hcell = bm.copy(self.hcell)
        NC = len(hcell) 
        isNotOK = bm.ones(NC, dtype=bm.bool)
        i = 0
        while bm.any(isNotOK):
            halfedge2cellnum[hcell[isNotOK]] = i
            hcell[isNotOK] = halfedge[hcell[isNotOK], 2]
            isNotOK1 = bm.copy(isNotOK)
            isNotOK[isNotOK1] = self.hcell[isNotOK1]!=hcell[isNotOK1]
            i += 1
        return halfedge2cellnum

    def halfedge_to_edge(self, index = _S):
        halfedge = self.halfedge
        hedge = self.hedge
        NE = self.NE

        halfedge2edge = bm.zeros(len(halfedge), dtype=self.itype)
        halfedge2edge[hedge] = bm.arange(NE, dtype=self.itype)
        halfedge2edge[halfedge[hedge, 4]] = bm.arange(NE, dtype=self.itype)
        return halfedge2edge[index] 


    ######################## 与插值点有关的接口 ######################
    def number_of_global_ipoints(self, p: int) -> int:
        """
        @brief 插值点总数
        """
        if self.NV==3:
            return self.NN + (p-1)*self.NE + (p-2)*(p-1)//2*self.NC
        else:
            gdof = self.NN 
            if p > 1:
                NE = self.number_of_edges()
                NC = self.number_of_cells()
                gdof += NE*(p-1) + NC*(p-1)*p//2
            return gdof

    def number_of_local_ipoints(self, p: int, iptype):
        """
        @brief 获取局部插值点的个数
        """
        if self.NV==3:
            if iptype in {'cell', 2}:
                return (p+1)*(p+2)//2
            elif iptype in {'face', 'edge',  1}: # 包括两个顶点
                return p + 1
            elif iptype in {'node', 0}:
                return 1
        else:
            if iptype in {'all'}:
                NV = self.number_of_vertices_of_cells()
                ldof = NV + (p-1)*NV + (p-1)*p//2
                return ldof
            elif iptype in {'cell', 2}:
                return (p-1)*p//2
            elif iptype in {'edge', 'face', 1}:
                return (p+1)
            elif iptype in {'node', 0}:
                return 1

    def edge_to_ipoint(self, p: int, index=_S) -> TensorLike:
        """
        @brief 获取网格边与插值点的对应关系

        """
        if isinstance(index, slice) and index == slice(None):
            NE = self.number_of_edges()
            index = bm.arange(NE)
        elif isinstance(index, bm.ndarray) and (index.dtype == bm.bool_):
            index, = bm.nonzero(index)
            NE = len(index)
        elif isinstance(index, list) and (type(index[0]) is bm.bool_):
            index, = bm.nonzero(index)
            NE = len(index)
        else:
            NE = len(index)

        NN = self.NN

        edge = self.edge_to_node()[index]
        edge2ipoints = bm.zeros((NE, p+1), dtype=self.itype)
        edge2ipoints[:, [0, -1]] = edge
        if p > 1:
            idx = NN + bm.arange(p-1)
            edge2ipoints[:, 1:-1] =  (p-1)*index[:, None] + idx
        return edge2ipoints



    def cell_to_ipoint(self, p: int, index=_S) -> TensorLike:
        """
        @brief
        """
        cell = self.cell_to_node()[index]
        if p == 1:
            return cell[index]
        else:
            if self.NV==3:
                if self.cell2ipt is None:
                    mi = self.multi_index_matrix(p=p, etype=2)
                    idx0, = bm.nonzero(mi[:, 0] == 0)
                    idx1, = bm.nonzero(mi[:, 1] == 0)
                    idx2, = bm.nonzero(mi[:, 2] == 0)

                    edge2cell = self.edge_to_cell()
                    NN = self.NN
                    NE = self.number_of_edges()
                    NC = self.number_of_cells()

                    e2p = self.edge_to_ipoint(p)
                    ldof = (p+1)*(p+2)//2
                    c2p = bm.zeros((NC, ldof), dtype=self.itype)

                    flag = edge2cell[:, 2] == 0
                    c2p[edge2cell[flag, 0][:, None], idx0] = e2p[flag]

                    flag = edge2cell[:, 2] == 1
                    c2p[edge2cell[flag, 0][:, None], bm.flip(idx1)] = e2p[flag]

                    flag = edge2cell[:, 2] == 2
                    c2p[edge2cell[flag, 0][:, None], idx2] = e2p[flag]


                    iflag = edge2cell[:, 0] != edge2cell[:, 1]

                    flag = iflag & (edge2cell[:, 3] == 0)
                    c2p[edge2cell[flag, 1][:, None], bm.flip(idx0)] = e2p[flag]

                    flag = iflag & (edge2cell[:, 3] == 1)
                    c2p[edge2cell[flag, 1][:, None], idx1] = e2p[flag]

                    flag = iflag & (edge2cell[:, 3] == 2)
                    c2p[edge2cell[flag, 1][:, None], bm.flip(idx2)] = e2p[flag]

                    cdof = (p-1)*(p-2)//2
                    flag = bm.sum(mi > 0, axis=1) == 3
                    c2p[:, flag] = NN + NE*(p-1) + bm.arange(NC*cdof,
                                                             dtype=self.itype).reshape(NC, cdof)
                    
                    self.cell2ipt = c2p
                    return c2p[index]
                else:
                    return self.cell2ipt[index]
            else:
                NC = self.number_of_cells()
                ldof = self.number_of_local_ipoints(p, iptype='all')

                location = bm.zeros(NC+1, dtype=self.itype)
                location[1:] = bm.cumsum(ldof, axis=0)

                cell2ipoint = bm.zeros(location[-1], dtype=self.itype)

                edge2ipoint = self.edge_to_ipoint(p)
                edge2cell = self.edge_to_cell()

                idx = location[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + bm.arange(p)
                cell2ipoint[idx] = edge2ipoint[:, 0:p]

                isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
                idx = (location[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p).reshape(-1, 1) + bm.arange(p)
                cell2ipoint[idx] = bm.flip(edge2ipoint[isInEdge, 1:p+1], axis=1)
               

                NN = len(self.hnode)
                NV = self.number_of_vertices_of_cells()
                NE = self.number_of_edges()
                cdof = self.number_of_local_ipoints(p, iptype='cell')
                idx = (location[:-1] + NV*p).reshape(-1, 1) + bm.arange(cdof)
                cell2ipoint[idx] = NN + NE*(p-1) + bm.arange(NC*cdof,dtype=self.itype).reshape(NC, cdof)
                return bm.split(cell2ipoint, location[1:-1], axis=0)[index]

    def boundary_node_flag(self):
        NN = self.NN
        halfedge =  self.halfedge 
        isBdHEdge = halfedge[:, 4]==bm.arange(self.NHE)
        isBdNode = bm.zeros(NN, dtype=bm.bool)
        isBdNode[halfedge[isBdHEdge, 0]] = True 
        return isBdNode

    def boundary_halfedge_flag(self):
        return self.halfedge[:, 4]==bm.arange(self.NHE)


    def nex_boundary_halfedge(self):
        halfedge = self.halfedge
        isBDHEdge = self.boundary_halfedge_flag()
        bdHEdge = bm.where(isBDHEdge)[0]
        nex = 100000*bm.ones(self.NHE, dtype=self.itype)
        nex[bdHEdge] = halfedge[bdHEdge, 2]

        isNotOK = bm.ones(bdHEdge.shape, dtype=bm.bool)
        while bm.any(isNotOK):
            nex[bdHEdge[isNotOK]] = halfedge[halfedge[nex[bdHEdge[isNotOK]], 4], 2]
            isNotOK = ~isBDHEdge[nex[bdHEdge]]
        return nex

    def boundary_edge_flag(self):
        halfedge =  self.halfedge
        hedge = self.hedge
        isBdEdge = hedge[:] == halfedge[hedge, 4] 
        return isBdEdge 

    boundary_face_flag = boundary_edge_flag

    def boundary_cell_flag(self):
        NC = self.NC
        halfedge =  self.halfedge
        isBdHEdge = halfedge[:, 4]==bm.arange(self.NHE)
        isBDCell = bm.zeros(NC, dtype=bm.bool)
        isBDCell[halfedge[isBdHEdge, 1]] = True 
        return isBDCell

    def boundary_node_index(self):
        NN = self.NN
        halfedge =  self.halfedge 
        isBdHEdge = halfedge[:, 4]==bm.arange(self.NHE)
        return halfedge[isBdHEdge, 0]

    def boundary_edge_index(self):
        isBdEdge = self.boundary_edge_flag()
        idx, = bm.nonzero(isBdEdge)
        return idx

    def boundary_halfedge_index(self):
        isBdHEdge = self.boundary_halfedge_flag()
        idx, = bm.nonzero(isBdHEdge)
        return idx

    boundary_face_index = boundary_edge_index

    def boundary_cell_index(self):
        NN = self.NN
        halfedge =  self.halfedge 
        isBdHEdge = halfedge[:, 4]==bm.arange(self.NHE)
        return halfedge[isBdHEdge, 1]








HalfEdgeMesh2d.set_ploter('polygon2d')
