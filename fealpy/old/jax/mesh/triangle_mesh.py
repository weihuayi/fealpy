
from typing import Optional, Union, List

import jax.numpy as jnp

from fealpy.jax.mesh.mesh_base import _S
from fealpy.jax.mesh.quadrature import Quadrature

from .. import logger
from . import functional as F
from .mesh_base import SimplexMesh, estr2dim
from .utils import Array
from jax import config

config.update("jax_enable_x64", True)

Index = Union[Array, int, slice]
_dtype = jnp.dtype

_S = slice(None)


class TriangleMesh(SimplexMesh):
    def __init__(self, node: Array, cell: Array) -> None:
        super().__init__(TD=2)
        # constant tensors
        kwargs = {'dtype': cell.dtype}
        self.cell = cell
        self.localEdge = jnp.array([(1, 2), (2, 0), (0, 1)], **kwargs)
        self.localFace = jnp.array([(1, 2), (2, 0), (0, 1)], **kwargs)
        self.ccw = jnp.array([0, 1, 2], **kwargs)

        self.localCell = jnp.array([
            (0, 1, 2),
            (1, 2, 0),
            (2, 0, 1)], **kwargs)

        self.construct()

        self.node = node
        self._attach_functionals()

    def _attach_functionals(self):
        GD = self.geo_dimension()
        if GD == 2:
            self._cell_area = F.simplex_measure
            self._grad_lambda = F.tri_grad_lambda_2d
        elif GD == 3:
            self._cell_area = F.tri_area_3d
            self._grad_lambda = F.tri_grad_lambda_3d
        else:
            logger.warn(f"{GD}D triangle mesh is not well supported: "
                        "cell_area and grad_lambda are not available. "
                        "Any operation involving them will fail.")

    # entity
    def entity_measure(self, etype: Union[int, str], index: Optional[Index]=None) -> Array:
        node = self.node
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        if etype == 0:
            return jnp.array([0,], dtype=self.ftype)
        elif etype == 1:
            edge = self.entity(1, index)
            return F.edge_length(edge, node)
        elif etype == 2:
            cell = self.entity(2, index)
            return self._cell_area(cell, node)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")

    # quadrature
    def quadrature_formula(self, q: int, etype: Union[int, str]='cell',
                           qtype: str='legendre') -> Quadrature: # TODO: other qtype
        from .quadrature import TriangleQuadrature
        from .quadrature import GaussLegendreQuadrature

        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        kwargs = {'dtype': self.ftype}
        if etype == 2:
            quad = TriangleQuadrature(q, **kwargs)
        elif etype == 1:
            quad = GaussLegendreQuadrature(q, **kwargs)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")

        return quad

    # ipoints
    def interpolation_points(self, p: int, index: Index=_S) -> Array:
        """Fetch all p-order interpolation points on the triangle mesh."""
        node = self.entity('node')
        if p == 1:
            return node
        if p <= 0:
            raise ValueError("p must be a integer larger than 0.")

        ipoint_list = []
        kwargs = {'dtype': self.ftype}

        GD = self.geo_dimension()
        ipoint_list.append(node) # ipoints[:NN, :]

        edge = self.entity('edge')
        w = jnp.zeros((p - 1, 2), **kwargs)
        w = w.at[:, 0].set(jnp.arange(p - 1, 0, -1, **kwargs)/p)
        w = w.at[:, 1].set(jnp.flip(w[:, 0], axis=0))
        ipoints_from_edge = jnp.einsum('ij, ...jm->...im', w,
                                         node[edge, :]).reshape(-1, GD) # ipoints[NN:NN + (p - 1) * NE, :]
        ipoint_list.append(ipoints_from_edge)

        if p >= 3:
            TD = self.top_dimension()
            cell = self.entity('cell')
            multiIndex = self.multi_index_matrix(p, TD)
            isEdgeIPoints = (multiIndex == 0)
            isInCellIPoints = ~(isEdgeIPoints[:, 0] | isEdgeIPoints[:, 1] |
                                isEdgeIPoints[:, 2])
            w = multiIndex[isInCellIPoints, :].astype(self.ftype)/p
            ipoints_from_cell = jnp.einsum('ij, kj...->ki...', w,
                                          node[cell, :]).reshape(-1, GD) # ipoints[NN + (p - 1) * NE:, :]
            ipoint_list.append(ipoints_from_cell)

        return jnp.concatenate(ipoint_list, axis=0)  # (gdof, GD)

    def cell_to_ipoint(self, p: int, index: jnp.ndarray=_S) -> jnp.ndarray:
        cell = self.cell
        if p == 1:
            return cell[index]

        mi = self.multi_index_matrix(p, 2)
        idx0, = jnp.nonzero(mi[:, 0] == 0)
        idx1, = jnp.nonzero(mi[:, 1] == 0)
        idx2, = jnp.nonzero(mi[:, 2] == 0)
        kwargs = {'dtype': self.itype}

        face2cell = self.face_to_cell()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()

        e2p = self.edge_to_ipoint(p)
        ldof = self.number_of_local_ipoints(p, 'cell')
        c2p = jnp.zeros((NC, ldof), **kwargs)

        flag = face2cell[:, 2] == 0
        c2p = c2p.at[tuple([face2cell[flag, 0][:, None], idx0])].set(e2p[flag])
        # c2p[face2cell[flag, 0][:, None], idx0] = e2p[flag]

        flag = face2cell[:, 2] == 1
        idx1_ = jnp.flip(idx1, axis=0)
        c2p = c2p.at[tuple([face2cell[flag, 0][:, None], idx1_])].set(e2p[flag])
        # c2p[face2cell[flag, 0][:, None], idx1_] = e2p[flag]

        flag = face2cell[:, 2] == 2
        c2p = c2p.at[tuple([face2cell[flag, 0][:, None], idx2])].set(e2p[flag])
        # c2p[face2cell[flag, 0][:, None], idx2] = e2p[flag]

        iflag = face2cell[:, 0] != face2cell[:, 1]

        flag = iflag & (face2cell[:, 3] == 0)
        idx0_ = jnp.flip(idx0, axis=0)
        c2p = c2p.at[tuple([face2cell[flag, 1][:, None], idx0_])].set(e2p[flag])
        # c2p[face2cell[flag, 1][:, None], idx0_] = e2p[flag]

        flag = iflag & (face2cell[:, 3] == 1)
        c2p = c2p.at[tuple([face2cell[flag, 1][:, None], idx1])].set(e2p[flag])
        # c2p[face2cell[flag, 1][:, None], idx1] = e2p[flag]

        flag = iflag & (face2cell[:, 3] == 2)
        idx2_ = jnp.flip(idx2, axis=0)
        c2p = c2p.at[tuple([face2cell[flag, 1][:, None], idx2_])].set(e2p[flag])
        # c2p[face2cell[flag, 1][:, None], idx2_] = e2p[flag]

        cdof = (p-1)*(p-2)//2
        flag = jnp.sum(mi > 0, axis=1) == 3
        c2p = c2p.at[:, flag].set(NN + NE*(p-1) + jnp.arange(NC*cdof, **kwargs).reshape(NC, cdof))
        # c2p[:, flag] = NN + NE*(p-1) + jnp.arange(NC*cdof, **kwargs).reshape(NC, cdof)
        return c2p[index]

    def face_to_ipoint(self, p: int, index: Index=_S) -> Array:
        return self.edge_to_ipoint(p, index)

    # shape function
    def grad_lambda(self, index: Index=_S):
        return self._grad_lambda(self.cell[index], self.node)

    # constructor
    @classmethod
    def from_box(cls, box: List[int]=[0, 1, 0, 1], nx=10, ny=10, threshold=None, *,
                 itype: Optional[_dtype]=jnp.int_,
                 ftype: Optional[_dtype]=jnp.float64):
        """Generate a uniform triangle mesh for a box domain.

        Parameters:
            box (List[int]): 4 integers, the left, right, bottom, top of the box.\n
            nx (int, optional): Number of divisions along the x-axis, defaults to 10.\n
            ny (int, optional): Number of divisions along the y-axis, defaults to 10.\n
            threshold (Callable | None, optional): Optional function to filter cells.
                Based on their barycenter coordinates, defaults to None.

        Returns:
            TriangleMesh: Triangle mesh instance.
        """
        fkwargs = {'dtype': ftype}
        ikwargs = {'dtype': itype}
        NN = (nx + 1) * (ny + 1)
        NC = nx * ny
        X, Y = jnp.meshgrid(
            jnp.linspace(box[0], box[1], nx + 1, **fkwargs),
            jnp.linspace(box[2], box[3], ny + 1, **fkwargs),
            indexing='ij'
        )
        node = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

        idx = jnp.arange(NN, **ikwargs).reshape(nx + 1, ny + 1)
        cell = jnp.zeros((2 * NC, 3), **ikwargs)
        cell = cell.at[:NC, 0].set(idx[1:, 0:-1].T.flatten())
        cell = cell.at[:NC, 1].set(idx[1:, 1:].T.flatten())
        cell = cell.at[:NC, 2].set(idx[0:-1, 0:-1].T.flatten())
        cell = cell.at[NC:, 0].set(idx[0:-1, 1:].T.flatten())
        cell = cell.at[NC:, 1].set(idx[0:-1, 0:-1].T.flatten())
        cell = cell.at[NC:, 2].set(idx[1:, 1:].T.flatten())

        if threshold is not None:
            bc = jnp.sum(node[cell, :], axis=1) / cell.shape[1]
            isDelCell = threshold(bc)
            cell = cell[~isDelCell]
            isValidNode = jnp.zeros(NN, dtype=jnp.bool_)
            isValidNode[cell] = True
            node = node[isValidNode]
            idxMap = jnp.zeros(NN, dtype=cell.dtype)
            idxMap[isValidNode] = range(isValidNode.sum())
            cell = idxMap[cell]

        return cls(node, cell)

    @classmethod
    def from_numpy(cls, mesh):
        import numpy as np

        new_mesh = cls.__new__(cls)
        SimplexMesh.__init__(new_mesh, TD=2)

        for name, tensor_obj in mesh.__dict__.items():
            if isinstance(tensor_obj, np.ndarray):
                setattr(new_mesh, name, jnp.array(tensor_obj))

        # NOTE: Meshes in old numpy version has `ds`` instead of `_entity_storage`.
        if hasattr(mesh, '_entity_storage'):
            for etype, entity in mesh._entity_storage.items():
                new_mesh._entity_storage[etype] = jnp.array(entity)

        if hasattr(mesh, 'ds'):
            for name, tensor_obj in mesh.ds.__dict__.items():
                if isinstance(tensor_obj, np.ndarray):
                    setattr(new_mesh, name, jnp.array(tensor_obj))

        new_mesh._attach_functionals()

        return new_mesh
