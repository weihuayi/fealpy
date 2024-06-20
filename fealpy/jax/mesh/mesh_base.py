
import numpy as np

from typing import (
    Literal, Callable, Optional, Union, TypeVar,
    overload, Dict, Any
)
from numpy.typing import NDArray

import jax
import jax.numpy as jnp

from . import functional as F
from .utils import Array, EntityName, Index, _int_func, _S, _T, _dtype, _device


class MeshDS():
    _STORAGE_ATTR = ['cell', 'face', 'edge', 'node']
    cell: Array 
    face: Array 
    edge: Array 
    node: Array 
    face2cell: Array 
    cell2edge: Array 
    localEdge: Array # only for homogeneous mesh
    localFace: Array # only for homogeneous mesh

    def __init__(self, NN: int, TD: int) -> None: 
        self._entity_storage: Dict[int, _T] = {}
        self.NN = NN
        self.TD = TD

    def construct(self) -> None:
        """
        """
        NC = self.cell.shape[0]
        NFC = self.cell.shape[1]

        totalFace = self.total_face()
        _, i0, j = jnp.unique(
            jnp.sort(totalFace, axis=1),
            return_index=True,
            return_inverse=True,
            axis=0
        )
        self.face = totalFace[i0, :]
        self.edge = self.face
        NF = i0.shape[0]

        i1 = np.zeros(NF, dtype=self.itype)
        i1[j.ravel()] = np.arange(3*NC, dtype=self.itype)

        self.cell2edge = j.reshape(NC, 3)
        self.cell2face = self.cell2edge
        self.face2cell = jnp.vstack([i0//3, i1//3, i0%3, i1%3]).T
        self.edge2cell = self.face2cell

        logger.info(f"Construct the mesh toplogy relation with {NF} edge (or face).")

class Mesh(MeshDS):
    @property
    def ftype(self) -> _dtype:
        node = self.entity(0)
        if node is None:
            raise RuntimeError('Can not get the float type as the node '
                               'has not been assigned.')
        return node.dtype

    def geo_dimension(self) -> int:
        node = self.entity(0)
        if node is None:
            raise RuntimeError('Can not get the geometrical dimension as the node '
                               'has not been assigned.')
        return node.shape[-1]

    GD = property(geo_dimension)

class HomogeneousMesh(Mesh):
    def interpolation_points(self, p: int, index: Index=_S) -> Tensor:
        raise NotImplementedError

    def cell_to_ipoint(self, p: int, index: Index=_S) -> Tensor:
        raise NotImplementedError

    def face_to_ipoint(self, p: int, index: Index=_S) -> Tensor:
        raise NotImplementedError

class SimplexMesh(HomogeneousMesh):
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell'):
        raise NotImplementedError

    def number_of_global_ipoints(self, p: int):
        raise NotImplementedError

    def grad_lambda(self, index: Index=_S) -> Tensor:
        raise NotImplementedError

    def shape_function(self, bc: Field, p: int=1, *, 
                       variable: str='u', mi: Optional[Field]=None) -> Field:
        raise NotImplementedError
        

