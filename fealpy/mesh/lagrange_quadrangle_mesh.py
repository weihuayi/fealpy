from typing import Union, Optional, Sequence, Tuple, Any

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .utils import estr2dim
from .mesh_base import TensorMesh
from .quadrangle_mesh import QuadrangleMesh

class LagrangeQuadrangleMesh(TensorMesh):
    def __init__(self, node: TensorLike, cell: TensorLike, p=1, surface=None,
            construct=False):
        super().__init__(TD=2, itype=cell.dtype, ftype=node.dtype)

        kwargs = bm.context(cell)
        GD = node.shape[1]
        self.p = p
        self.cell = cell
        self.surface = surface

        if p == 1:
            self.node = node
        else:
            NN = self.number_of_nodes()
            self.node = bm.zeros((NN, self.GD), dtype=bm.float64)
            bc = bm.multi_index_matrix(p, TD)/p
            bc = bm.einsum('im, jn -> ijmn', bc, bc).reshape(-1, 4)
            self.node[self.cell] = bm.einsum('ijn, kj -> ikn', node[cell], bc)

        #self.localEdge = self.generate_local_lagrange_edges(p)
        #self.localFace = self.localEdge
        self.ccw = bm.tensor([0, 2, 3, 1], **kwargs)

        if construct:
            self.construct()

        self.meshtype = 'lquad'
        self.qmesh = None # 网格的顶点必须在球面上

        self.nodedata = {}
        self.edgedata = {}
        self.celldata = {}
        self.meshdata = {}

    def reference_cell_measure(self):
        return 1
    
    def generate_local_lagrange_edges(self, p: int) -> TensorLike:
        """
        Generate the local edges for Lagrange elements of order p.
        """
        multiIndex = self.multi_index_matrix(p, 1, dtype=self.ftype, device=bm.get_device(cell))
        #multiIndex = bm.multi_index_matrix(p, TD)

        localEdge = bm.zeros((4, p+1), dtype=bm.int32)

        return localEdge
    
    def interpolation_points(self, p: int, index: Index=_S):
        """Fetch all p-order interpolation points on the quadrangle mesh."""
        pass

    @classmethod
    def from_quadrangle_mesh(cls, mesh, p: int, surface=None):
        bnode = mesh.entity('node')
        node = mesh.interpolation_points(p)
        cell = mesh.cell_to_ipoint(p)
        if surface is not None:
            bnode[:],_ = surface.project(bnode)
            node,_ = surface.project(node)

        lmesh = cls(node, cell, p=p, construct=True)
        lmesh.qmesh = mesh

        lmesh.edge2cell = mesh.edge2cell # (NF, 4)
        lmesh.cell2edge = mesh.cell_to_edge()
        lmesh.edge  = mesh.edge_to_ipoint(p)
        return lmesh

    # quadrature
    def quadrature_formula(self, q, etype: Union[int, str] = 'cell'):
        from ..quadrature import GaussLegendreQuadrature, TensorProductQuadrature
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        qf = GaussLegendreQuadrature(q, dtype=self.ftype, device=self.device)
        if etype == 2:
            return TensorProductQuadrature((qf, qf))
        elif etype == 1:
            return TensorProductQuadrature((qf, ))
        else:
            raise ValueError(f"entity type: {etype} is wrong!")

    def bc_to_point(self, bc: TensorLike, index: Index=_S, etype='cell'):
        node = self.node
        TD = len(bc)
        phi = self.shape_function(bc)
        p = bm.einsum()
        return p

    # shape function
    def shape_function(self, bc: TensorLike, p=None):
        """
        @berif 
        bc 是一个长度为 TD 的 tuple 数组
        bc[i] 是一个一维积分公式的重心坐标数组
        假设 bc[0]==bc[1]== ... ==bc[TD-1]
        """
        p = self.p if p is None else p
        TD = len(bc)
        phi = bm.simplex_shape_function(bc[0], p)
        if TD == 2:
            phi = bm.einsum('im, jn -> ijmn', phi, phi) #TODo
            shape = phi.reshape[:-2] + (-1,)
            phi = phi.reshape(shape) # 展平自由度
            shape = (-1, 1) + phi.shape[-1:] # 增加一个单元轴，方便广播运算
            phi = phi.reshape(shape) # 展平积分点
        return phi

    def grad_shape_function(self, bc: TensorLike, p=None, 
            index: Index=_S, variables='x'):
        pass

    def vtk_cell_type(self, etype='cell'):
        """
        @berif  返回网格单元对应的 vtk类型。
        """
        if etype in {'cell', 2}:
            VTK_LAGRANGE_TRIANGLE = 69
            return VTK_LAGRANGE_TRIANGLE 
        elif etype in {'face', 'edge', 1}:
            VTK_LAGRANGE_CURVE = 68
            return VTK_LAGRANGE_CURVE

    def to_vtk(self, etype='cell', index: Index=_S, fname=None):
        """
        Parameters
        ----------

        @berif 把网格转化为 VTK 的格式
        """
        from fealpy.mesh.vtk_extent import vtk_cell_index, write_to_vtu

        node = self.entity('node')
        GD = self.geo_dimension()
        if GD == 2:
            node = bm.concatenate((node, bm.zeros((node.shape[0], 1), dtype=bm.float64)), axis=1)

        #cell = self.entity(etype)[index]
        cell = self.entity(etype, index)
        cellType = self.vtk_cell_type(etype)
        idx = vtk_cell_index(self.p, cellType)
        NV = cell.shape[-1]

        cell = bm.concatenate((bm.zeros((len(cell), 1), dtype=cell.dtype), cell[:, idx]), axis=1)
        cell[:, 0] = NV

        NC = len(cell)
        if fname is None:
            return node, cell.flatten(), cellType, NC 
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                    nodedata=self.nodedata,
                    celldata=self.celldata)
