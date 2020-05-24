import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, tril, triu
from ..quadrature import TriangleQuadrature, QuadrangleQuadrature, GaussLegendreQuadrature 
from .Mesh2d import Mesh2d
from .adaptive_tools import mark
from .mesh_tools import show_halfedge_mesh
from ..common.Tools import hash2map


# subdomain: 单元所处的子区域的标记编号
#  0: 表示外部无界区域
# -n: n >= 1, 表示编号为 -n 洞
#  n: n >= 1, 表示编号为  n 的内部子区域

class HalfEdgeMesh2d(Mesh2d):
    def __init__(self, node, halfedge, subdomain, NV=None, nodedof=None):
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
        """

        self.itype = halfedge.dtype
        self.ftype = node.dtype

        self.node = node
        self.ds = HalfEdgeMesh2dDataStructure(node.shape[0], 
                subdomain, halfedge, NV=NV)
        self.meshtype = 'halfedge'

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


    def init_level_info(self):
        NN = self.number_of_nodes()
        NE = self.number_of_edges()

        NC = self.number_of_all_cells() # 实际单元个数
        self.celldata['level'] = np.zeros(NC, dtype=self.itype)
        self.halfedgedata['level'] = np.zeros(2*NE, dtype=self.itype)
        self.nodedata['level'] = np.zeros(NN, dtype=self.itype)

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
            

            halfedge = np.zeros((2*NE, 6), dtype=mesh.itype)
            halfedge[:, 0] = edge.flat

            halfedge[0::2, 1][isInEdge] = edge2cell[isInEdge, 1] + 1
            halfedge[1::2, 1] = edge2cell[:, 0] + 1

            halfedge[0::2, 4] = range(1, 2*NE, 2)
            halfedge[1::2, 4] = range(0, 2*NE, 2)
            halfedge[1::2, 5]  = 1

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

            return cls(node, subdomain, halfedge)
        else:
            newMesh =  cls(mesh.node, mesh.subdomain, mesh.ds.halfedge.copy())
            newMesh.celldata['level'][:] = mesh.celldata['level']
            newMesh.nodedata['level'][:] = mesh.nodedata['level']
            newMesh.halfedge['level'][:] = mesh.halfedgedata['level']
            return newMesh

    def entity(self, etype=2):
        if etype in {'cell', 2}:
            return self.ds.cell_to_node()
        elif etype in {'edge', 'face', 1}:
            return self.ds.edge_to_node()
        elif etype in {'halfedge'}:
            return self.ds.halfedge
        elif etype in {'node', 0}:
            return self.node
        else:
            raise ValueError("`entitytype` is wrong!")

    def entity_barycenter(self, etype='cell', index=None):
        node = self.node
        dim = self.geo_dimension()
        if etype in {'cell', 2}:
            cell2node = self.ds.cell_to_node(return_sparse=True)
            NV = self.ds.number_of_vertices_of_cells().reshape(-1,1)
            bc = cell2node*node/NV
        elif etype in {'edge', 'face', 1}:
            edge = self.ds.edge_to_node()
            bc = np.sum(node[edge, :], axis=1).reshape(-1, dim)/edge.shape[1]
        elif etype in {'node', 1}:
            bc = node
        return bc

    def node_normal(self):
        node = self.node
        cell, cellLocation = self.entity('cell') # TODO: for tri and quad case
        idx1 = np.zeros(cell.shape[0], dtype=np.int)
        idx2 = np.zeros(cell.shape[0], dtype=np.int)

        idx1[0:-1] = cell[1:]
        idx1[cellLocation[1:]-1] = cell[cellLocation[:-1]]
        idx2[1:] = cell[0:-1]
        idx2[cellLocation[:-1]] = cell[cellLocation[1:]-1]
        w = np.array([(0,-1),(1,0)])
        d = node[idx1] - node[idx2]
        return 0.5*d@w

    def cell_area(self, index=None):
        NC = self.number_of_cells()
        node = self.entity('node')
        halfedge = self.ds.halfedge
        hflag = self.ds.hflag
        cidxmap = self.ds.cidxmap

        e0 = halfedge[halfedge[hflag, 3], 0]
        e1 = halfedge[hflag, 0]

        w = np.array([[0, -1], [1, 0]], dtype=np.int)
        v= (node[e1] - node[e0])@w
        val = np.sum(v*node[e0], axis=1)

        a = np.zeros(NC, dtype=self.ftype)
        np.add.at(a, cidxmap[halfedge[hflag, 1]], val)
        a /=2
        return a

    def cell_barycenter(self, return_all=False):
        GD = self.geo_dimension()
        node = self.entity('node')
        halfedge = self.ds.halfedge
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
            hflag = self.ds.hflag
            cidxmap = self.ds.cidxmap
            e0 = halfedge[halfedge[hflag, 3], 0]
            e1 = halfedge[hflag, 0]
            w = np.array([[0, -1], [1, 0]], dtype=np.int)
            v= (node[e1] - node[e0])@w
            val = np.sum(v*node[e0], axis=1)
            ec = val.reshape(-1, 1)*(node[e1]+node[e0])/2
            a = np.zeros(NC, dtype=self.ftype)
            c = np.zeros((NC, GD), dtype=self.ftype)
            np.add.at(a, cidxmap[halfedge[hflag, 1]], val)
            np.add.at(c, (cidxmap[halfedge[hflag, 1]], np.s_[:]), ec)
            a /=2
            c /=3*a.reshape(-1, 1)
            return c

    def edge_bc_to_point(self, bcs, index=None):
        node = self.entity('node')
        edge = self.entity('edge')
        index = index if index is not None else np.s_[:]
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge[index]])
        return ps

    def add_halfedge_plot(self, axes,
        index=None, showindex=False,
        nodecolor='r', edgecolor=['r', 'k'], markersize=20,
        fontsize=8, fontcolor='k', multiindex=None, linewidth=0.5):

        show_halfedge_mesh(axes, self,
                index=index, showindex=showindex,
                nodecolor=nodecolor, edgecolor=edgecolor, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor, 
                multiindex=multiindex, linewidth=linewidth)

    def print(self):
        cell, cellLocation = self.entity('cell')
        print("cell:\n", cell)
        print("cellLocation:\n", cellLocation)
        print("cell2edge:\n", self.ds.cell_to_edge(return_sparse=False))
        print("cell2hedge:\n")
        for i, val in enumerate(self.ds.cell2hedge[:-1]):
            print(i, ':', val)

        print("edge:")
        for i, val in enumerate(self.entity('edge')):
            print(i, ":", val)

class HalfEdgeMesh2dDataStructure():

    def __init__(self, NN, subdomain, halfedge, NV=None):
        self.init(NN, subdomain, halfedge, NV)

    def reinit(self, NN, subdomain, halfedge, NV=None):
        self.init(NN, subdomain, halfedge, NV)

    def init(self, NN, subdomain, halfedge, NV=None):

        self.itype = halfedge.dtype

        self.NN = NN
        self.NE = len(halfedge)//2
        self.NF = self.NE
        self.subdomain = subdomain 

        # 区域内部的单元标记, 这里默认排前面的都是洞, 或者外部无界区域.
        idx, = np.nonzero(subdomain == 0)
        if len(idx) == 0:
            self.cellstart = 0 
        elif len(idx) == 1:
            self.cellstart = idx[0] + 1
        else:
            raise ValueError("The number of unbounded doamin is bigger than 1!")

        self.NC = len(subdomain) - self.cellstart # 区域内单元的个数
        self.hflag = subdomain[halfedge[:, 1]] > 0 # 所属单元是区域内部单元的半边标记

        NC = len(subdomain) # 实际单元个数, 包括外部无界区域和洞
        self.cidxmap = -np.ones(NC, dtype=self.itype) # 单元指标映射数组
        self.cidxmap[self.cellstart:] = range(self.NC)
        self.halfedge = halfedge
        
        self.cell2hedge = np.zeros(NC, dtype=self.itype)   # 存储每个单元的起始半边
        self.cell2hedge[halfedge[:, 1]] = range(2*self.NE) # 的编号

        if NV is None:
            NC = self.NC
            halfedge = self.halfedge
            hflag = self.hflag
            cidxmap = self.cidxmap
            self.NV = np.zeros(NC, dtype=self.itype)
            np.add.at(self.NV, cidxmap[halfedge[hflag, 1]], 1)
        else:
            assert NV == 3 or NV == 4
            self.NV = NV

        NHE = len(halfedge)
        edge = np.zeros((2*NHE, 2), dtype=facets.dtype)
        edge[:NHE] = halfedge[:, 0:2]
        edge[NHE:, 0] = halfedge[halfedge[:, 4], 0]
        edge[NHE:, 1] = halfedge[:, 1]
        idx = np.lexsort((edge[:, 0], edge[:, 1])).reshape(-1, 2)
        idx[:, 1] -= NHE

        self.nex = np.zero(NHE, dtype=self.itype) 
        self.pre = np.zero(NHE, dtype=self.itype)

        self.nex[idx[:, 0]] = idx[:, 1]
        self.pre[self.nex] = range(NHE)


    def number_of_all_cells(self):
        return len(self.subdomain)

    def number_of_vertices_of_all_cells(self):
        NC = self.number_of_all_cells() 
        halfedge = self.halfedge
        NV = np.zeros(NC, dtype=self.itype)
        np.add.at(NV, halfedge[:, 1], 1)
        return NV

    def number_of_vertices_of_cells(self):
        return self.NV

    def number_of_nodes_of_cells(self):
        return self.NV

    def number_of_edges_of_cells(self):
        return self.NV

    def number_of_faces_of_cells(self):
        return self.NV

    def cell_to_node(self, return_sparse=False):
        NN = self.NN
        NC = self.NC
        halfedge = self.halfedge
        hflag = self.hflag
        cstart = self.cellstart
        cidxmap = self.cidxmap

        if return_sparse:
            val = np.ones(hflag.sum(), dtype=np.bool)
            I = cidxmap[halfedge[hflag, 1]]
            J = halfedge[hflag, 0]
            cell2node = csr_matrix((val, (I, J)), shape=(NC, NN), dtype=np.bool)
            return cell2node
        elif type(self.NV) is np.ndarray: # polygon mesh
            cellLocation = np.zeros(NC+1, dtype=self.itype)
            cellLocation[1:] = np.cumsum(self.NV)
            cell2node = np.zeros(cellLocation[-1], dtype=self.itype)
            current = self.cell2hedge[cstart:]
            idx = cellLocation[:-1].copy()
            cell2node[idx] = halfedge[current, 0]
            NV0 = np.ones(NC, dtype=self.itype)
            isNotOK = NV0 < self.NV
            while isNotOK.sum() > 0:
               current[isNotOK] = halfedge[current[isNotOK], 2]
               idx[isNotOK] += 1
               NV0[isNotOK] += 1
               cell2node[idx[isNotOK]] = halfedge[current[isNotOK], 0]
               isNotOK = (NV0 < self.NV)
            return cell2node, cellLocation
        elif self.NV == 3: # tri mesh
            cell2node = np.zeros(NC, 3)
            current = self.cell2hedge[cstart:]
            cell2node[:, 0] = halfedge[current, 0]
            current = halfedge[current, 2]
            cell2node[:, 1] = halfedge[current, 0]
            current = halfedge[current, 2]
            cell2node[:, 2] = halfedge[current, 0]
            return cell2node
        elif self.NV == 4: # quad mesh
            cell2node = np.zeros(NC, 3)
            current = self.cell2hedge[cstart:]
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
        hflag = self.hflag

        J = np.zeros(2*NE, dtype=self.itype)
        isMainHEdge = (halfedge[:, 5] == 1)
        J[isMainHEdge] = range(NE)
        J[halfedge[isMainHEdge, 4]] = range(NE)
        if return_sparse:
            val = np.ones(2*NE, dtype=np.bool)
            cell2edge = csr_matrix((val[hflag], (halfedge[hflag, 1],
                J[hflag])), shape=(NC, NE), dtype=np.bool)
            return cell2edge
        elif type(self.NV) is np.ndarray:

            cellLocation = np.zeros(NC+1, dtype=self.itype)
            cellLocation[1:] = np.cumsum(self.NV)

            cell2edge = np.zeros(cellLocation[-1], dtype=self.itype)
            current = halfedge[self.cell2hedge[cstart:], 2] # 下一个边
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
            return cell2edge
        elif self.NV == 3: # tri mesh
            cell2edge = np.zeros(NC, 3)
            current = self.cell2hedge[cstart:]
            cell2edge[:, 1] = J[current]
            cell2edge[:, 2] = J[halfedge[current, 2]]
            cell2edge[:, 0] = J[halfedge[current, 3]]
            return cell2edge
        elif self.NV == 4: # quad mesh
            cell2edge = np.zeros(NC, 4)
            current = self.cell2hedge[cstart:]
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
        hflag = self.hflag
        cidxmap = self.cidxmap
        if return_sparse:
            flag = hflag & hflag[halfedge[:, 4]]
            val = np.ones(flag.sum(), dtype=np.bool)
            I = halfedge[flag, 1]
            J = halfedge[halfedge[flag, 4], 1]
            cell2cell = coo_matrix((val, (I, J)), shape=(NC, NC), dtype=np.bool)
            cell2cell+= coo_matrix((val, (J, I)), shape=(NC, NC), dtype=np.bool)
            return cell2cell.tocsr()
        elif type(self.NV) is np.ndarray:
            cellLocation = np.zeros(NC+1, dtype=self.itype)
            cellLocation[1:] = np.cumsum(self.NV)
            cell2cell = np.zeros(cellLocation[-1], dtype=self.itype)
            current = halfedge[self.cell2hedge[cstart:], 2] # 下一个边
            idx = cellLocation[:-1]
            cell2cell[idx] = cidxmap[halfedge[halfedge[current, 4], 1]]
            NV0 = np.ones(NC, dtype=self.itype)
            isNotOK = NV0 < self.NV
            while isNotOK.sum() > 0:
                current[isNotOK] = halfedge[current[isNotOK], 2]
                idx[isNotOK] += 1
                NV0[isNotOK] += 1
                cell2cell[idx[isNotOK]] = cidxmap[halfedge[halfedge[current[isNotOK], 4], 1]]
                isNotOK = (NV0 < self.NV)
            idx = np.repeat(range(NC), self.NV)
            flag = (cell2cell == -1)
            cell2cell[flag] = idx[flag]
            return cell2cell
        elif self.NV == 3: # tri mesh
            cell2cell = np.zeros(NC, 3)
            current = self.cell2hedge[cstart:]
            cell2cell[:, 1] = cidxmap[halfedge[halfedge[current, 4], 1]]
            cell2cell[:, 2] = cidxmap[halfedge[halfedge[halfedge[current, 2], 4], 1]]
            cell2cell[:, 0] = cidxmap[halfedge[halfedge[halfedge[current, 3], 4], 1]]
            idx = np.repeat(range(NC), 3).reshape(NC, 3)
            flag = (cell2cell == -1)
            cell2cell[flag] = idx[flag]
            return cell2cella
        elif self.NV == 4: # quad mesh
            cell2cell = np.zeros(NC, 4)
            current = self.cell2hedge[cstart:]
            cell2cell[:, 3] = cidxmap[halfedge[halfedge[current, 4], 1]] 
            current = halfedge[current, 2]
            cell2cell[:, 0] = cidxmap[halfedge[halfedge[current, 4], 1]] 
            current = halfedge[current, 2]
            cell2cell[:, 1] = cidxmap[halfedge[halfedge[current, 4], 1]] 
            current = halfedge[current, 2]
            cell2cell[:, 2] = cidxmap[halfedge[halfedge[current, 4], 1]]
            idx = np.repeat(range(NC), 4).reshape(NC, 4)
            flag = (cell2cell == -1)
            cell2cell[flag] = idx[flag]
            return cell2cell
        else:
            raise ValueError('The property NV should be None, 3 or 4! But the NV is {}'.format(self.NV))

    def edge_to_node(self, return_sparse=False):
        NN = self.NN
        NE = self.NE
        halfedge = self.halfedge
        isMainHEdge = halfedge[:, 5] == 1
        if return_sparse == False:
            edge = np.zeros((NE, 2), dtype=self.itype)
            edge[:, 0] = halfedge[halfedge[isMainHEdge, 4], 0]
            edge[:, 1] = halfedge[isMainHEdge, 0]
            return edge
        else:
            val = np.ones(NE, dtype=np.bool)
            edge2node = coo_matrix((val, (range(NE), halfedge[isMainHEdge,0])), shape=(NE, NN), dtype=np.bool)
            edge2node+= coo_matrix((val, (range(NE), halfedge[halfedge[isMainHEdge, 4], 0])), shape=(NE, NN), dtype=np.bool)
            return edge2node.tocsr()

    def edge_to_edge(self):
        edge2node = self.edge_to_node()
        return edge2node*edge2node.tranpose()

    def edge_to_cell(self):
        NE = self.NE
        NC = self.NC
        halfedge = self.halfedge
        cstart = self.cellstart
        cidxmap = self.cidxmap

        J = np.zeros(2*NE, dtype=self.itype)
        isMainHEdge = (halfedge[:, 5] == 1)
        J[isMainHEdge] = range(NE)
        J[halfedge[isMainHEdge, 4]] = range(NE)
        edge2cell = np.full((NE, 4), -1, dtype=self.itype)
        edge2cell[J[isMainHEdge], 0] = cidxmap[halfedge[isMainHEdge, 1]]
        edge2cell[J[halfedge[isMainHEdge, 4]], 1] = cidxmap[halfedge[halfedge[isMainHEdge, 4], 1]]
        if type(self.NV) is np.ndarray:
            current = halfedge[self.cell2hedge[cstart], 2] # 下一个边
            end = current.copy()
            lidx = np.zeros_like(current)
            isNotOK = np.ones_like(current, dtype=np.bool)
            while np.any(isNotOK):
                idx = J[current[isNotOK]]
                flag = (halfedge[current[isNotOK], 5] == 1)
                edge2cell[idx[flag], 2] = lidx[isNotOK][flag]
                edge2cell[idx[~flag], 3] = lidx[isNotOK][~flag]
                current[isNotOK] = halfedge[current[isNotOK], 2]
                lidx[isNotOK] += 1
                isNotOK = (current != end)
        elif self.NV == 3:
            current = self.cell2hedge[cstart]
            idx = J[current]
            flag = halfedge[current, 5] == 1 
            edge2cell[idx[flag], 2] = 1
            edge2cell[idx[~flag], 3] = 1

            idx = J[halfedge[current, 2]]
            flag = halfedge[halfedge[current, 2], 5] == 1 
            edge2cell[idx[flag], 2] = 2
            edge2cell[idx[~flag], 3] = 2

            idx = J[halfedge[current, 3]]
            flag = halfedge[halfedge[current, 3], 5] == 1 
            edge2cell[idx[flag], 2] = 0
            edge2cell[idx[~flag], 3] = 0
        elif self.NV == 4:
            current = self.cell2hedge[cstart]
            idx = J[current]
            flag = halfedge[current, 5] == 1 
            edge2cell[idx[flag], 2] = 3
            edge2cell[idx[~flag], 3] = 3

            current = halfedge[current, 2]
            idx = J[current]
            flag = halfedge[current, 5] == 1 
            edge2cell[idx[flag], 2] = 0
            edge2cell[idx[~flag], 3] = 0

            current = halfedge[current, 2]
            idx = J[current]
            flag = halfedge[current, 5] == 1 
            edge2cell[idx[flag], 2] = 1
            edge2cell[idx[~flag], 3] = 1

            current = halfedge[current, 2]
            idx = J[current]
            flag = halfedge[current, 5] == 1 
            edge2cell[idx[flag], 2] = 2
            edge2cell[idx[~flag], 3] = 2
        else:
            raise ValueError('The property NV should be None, 3 or 4! But the NV is {}'.format(self.NV))

        flag = edge2cell[:, 1] == -1
        edge2cell[flag, 1] = edge2cell[flag, 0]
        edge2cell[flag, 3] = edge2cell[flag, 2]
        return edge2cell

    def node_to_node(self):
        NN = self.NN
        NE = self.NE
        edge = self.edge_to_node()
        I = edge[:, 0:2].flat
        J = edge[:, 1::-1].flat
        val = np.ones(2*NE, dtype=np.bool)
        node2node = csr_matrix((val, (I, J)), shape=(NN, NN), dtype=np.bool)
        return node2node

    def node_to_cell(self, return_sparse=True):
        NN = self.NN
        NC = self.NC
        halfedge =  self.halfedge
        hflag = self.hflag
        cidxmap = self.cidxmap

        val = np.ones(hflag.sum(), dtype=np.bool)
        I = halfedge[hflag, 0]
        J = cidxmap[halfedge[hflag, 1]]
        node2cell = csr_matrix((val, (I.flat, J.flat)), shape=(NN, NC), dtype=np.bool)
        return node2cell

    def boundary_node_flag(self):
        NN = self.NN
        halfedge =  self.halfedge
        hflag = self.hflag
        isBdHEdge = hflag & (~hflag[halfedge[:, 4]])
        isBdNode = np.zeros(NN, dtype=np.bool)
        isBdNode[halfedge[isBdHEdge, 0]] = True 
        return isBdNode

    def boundary_edge_flag(self):
        NE = self.NE
        halfedge =  self.halfedge
        hflag = self.hflag
        isBdHEdge = hflag & (~hflag[halfedge[:, 4]])

        J = np.zeros(2*NE, dtype=self.itype)
        isMainHEdge = (halfedge[:, 5] == 1)
        J[isMainHEdge] = range(NE)
        return J[isBdHEdge] 

    def boundary_edge(self):
        edge = self.edge_to_node()
        return edge[self.boundary_edge_index()]

    def boundary_cell_flag(self):
        NC = self.NC
        halfedge =  self.halfedge
        hflag = self.hflag
        isBdHEdge = hflag & (~hflag[halfedge[:, 4]])

        isBdCell = np.zeros(NC, dtype=np.bool)
        idx = cidxmap[halfedge[isBdHEdge, 1]]
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
