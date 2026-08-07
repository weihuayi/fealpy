from __future__ import annotations

__all__ = ["distribute_space", "DistSpaceResult"]

from typing import TypeVar, Any, NamedTuple, Generic

from mpi4py.MPI import Comm, COMM_WORLD
from fealpy.functionspace import FunctionSpace

from ..backend import bm
from . import entity_mpi as _de
from . import distributed_mesh as _dm

_ST_co = TypeVar('_ST_co', bound='FunctionSpace')
_S = slice(None)


class DistSpaceResult(NamedTuple, Generic[_ST_co]):
    space: _ST_co
    dofcomm: _de.EntityMPI


def _mk_distributed_space_type(kind: type[_ST_co], /) -> type[_ST_co]:
    def __repr__(self: _ST_co) -> str:
        return f'<Distributed{kind.__name__} object at {hex(id(self))}>'

    def cell_to_dof(self, index=_S):
        return self.cell2dof[index]

    def face_to_dof(self, index=_S):
        return self.face2dof[index]

    def edge_to_dof(self, index=_S):
        return self.edge2dof[index]

    class_namespace = {
        "__repr__": __repr__,
        "cell_to_dof": cell_to_dof,
        "face_to_dof": face_to_dof,
        "edge_to_dof": edge_to_dof
    }
    return type(f'Dist{kind.__name__}', (kind,), class_namespace) # type: ignore


def distribute_space(
    space: _ST_co | None,
    distributed_mesh: _dm.DistMeshResult,
    *,
    root: int = 0,
    comm: Comm | None = None
) -> DistSpaceResult[_ST_co]:
    """Create distributed spaces."""
    if comm is None:
        comm = COMM_WORLD

    pmesh, mcomm = distributed_mesh
    root_entity = mcomm.entities[mcomm.root_entity_name]
    face_entity_name = pmesh.Entity("face:0").sector.schema_name
    face_entity = mcomm.entities[face_entity_name]
    edge_entity = mcomm.entities["segment"]
    all_cell_global_indices = comm.gather(root_entity._global_indices, root=root)
    all_face_global_indices = comm.gather(face_entity._global_indices, root=root)
    all_edge_global_indices = comm.gather(edge_entity._global_indices, root=root)

    lcell2dof = None
    lface2dof = None
    ledge2dof = None
    gdata: dict[str, Any] = {}

    if comm.Get_rank() == root:
        assert space is not None, 'root: The global space must be provided when root.'
        assert all_cell_global_indices is not None
        assert all_face_global_indices is not None
        assert all_edge_global_indices is not None

        cell2dof = space.cell_to_dof()
        face2dof = space.face_to_dof()
        edge2dof = space.edge_to_dof()  # type: ignore[attr-defined]
        lcell2dof = [bm.asarray(cell2dof[m], copy=True) for m in all_cell_global_indices]
        lface2dof = [bm.asarray(face2dof[m], copy=True) for m in all_face_global_indices]
        ledge2dof = [bm.asarray(edge2dof[m], copy=True) for m in all_edge_global_indices]

        gdata = {
            "space_type": type(space),
            "p": getattr(space, "p"),
            "NDOF": space.number_of_global_dofs(),
        }

    gdata = comm.bcast(gdata, root=root)
    lcell2dof = comm.scatter(lcell2dof, root)
    lface2dof = comm.scatter(lface2dof, root)
    ledge2dof = comm.scatter(ledge2dof, root)
    dof_mask = bm.zeros((gdata["NDOF"],), dtype=bm.bool)
    dof_mask[lcell2dof] = True
    dof_masks = comm.alltoall([dof_mask]*comm.Get_size())
    dofcomm = _de.dist_from_masks(dof_masks)

    space_type = _mk_distributed_space_type(gdata["space_type"])
    pspace = space_type(pmesh.fealpy_api(), gdata["p"])

    local_index = _dm._make_local_index(gdata["NDOF"], dof_mask)
    pspace.cell2dof = bm.asarray(local_index[lcell2dof], copy=True)
    pspace.face2dof = bm.asarray(local_index[lface2dof], copy=True)
    pspace.edge2dof = bm.asarray(local_index[ledge2dof], copy=True)

    return DistSpaceResult(pspace, dofcomm)
