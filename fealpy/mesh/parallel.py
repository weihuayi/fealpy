
from typing import Union, Dict, List, Tuple, Optional
from typing import TypeVar, Generic
from enum import IntEnum

from mpi4py.MPI import Comm, COMM_WORLD

from ..backend import backend_manager as bm
from ..backend import TensorLike as _DT
from ..typing import Index, _S
from .mesh_base import Mesh

_MT = TypeVar('_MT', bound=Mesh)

def split_homogeneous_mesh(mesh: _MT, /, masks: Tuple[_DT, ...]):
    # global boundary
    global_bdry_flags: Dict[str, _DT] = {}
    global_bdry_flags['node'] = mesh.boundary_node_flag()
    global_bdry_flags['cell'] = mesh.boundary_cell_flag()
    if mesh.TD >= 2:
        global_bdry_flags['face'] = mesh.boundary_face_flag()

    # source id and virtual table
    src_id: Dict[str, _DT] = {}
    local_id: Dict[str, _DT] = {}
    for etype in ('node', 'cell'):
        num = mesh.count(etype)
        src_id[etype] = bm.zeros((num,), dtype=bm.uint8, device=mesh.device)
        local_id[etype] = bm.zeros((num,), dtype=mesh.itype, device=mesh.device)

    MESHTYPE = mesh.__class__
    enumerated = list(enumerate(masks))
    flags_rev: List[Dict[str, _DT]] = []
    local_mesh_rev: List[_MT] = []
    NN = mesh.number_of_nodes()
    node = mesh.node
    cell = mesh.cell

    for idx, flag in reversed(enumerated):
        # split nodes
        src_id['node'] = bm.set_at(src_id['node'], cell[flag, :].reshape(-1), idx)

        node_flag = bm.zeros((NN,), dtype=bm.bool, device=mesh.device)
        node_flag = bm.set_at(node_flag, cell[flag, :], True)
        NN_LOCAL = bm.sum(node_flag)
        local_id['node'] = bm.set_at(
            local_id['node'], node_flag,
            bm.arange(NN_LOCAL, dtype=mesh.itype, device=mesh.device)
        )
        flags_rev.append({'node': node_flag})

        # split cells
        flags_rev[-1]['cell'] = flag
        src_id['cell'] = bm.set_at(src_id['cell'], flag, idx)
        if (flag.dtype == bm.bool):
            NC_LOCAL = bm.sum(flag)
        else:
            NC_LOCAL = flag.shape[0]
        local_id['cell'] = bm.set_at(
            local_id['cell'], flag,
            bm.arange(NC_LOCAL, dtype=mesh.itype, device=mesh.device)
        )

        # split mesh obj
        local_mesh = MESHTYPE(node[node_flag, :], local_id['node'][cell[flag]])
        local_mesh_rev.append(local_mesh)

    for flags, local_mesh in zip(reversed(flags_rev), reversed(local_mesh_rev)):
        global_flags_on_bdry: Dict[str, _DT] = {}
        src_id_on_bdry: Dict[str, _DT] = {}
        local_id_on_bdry: Dict[str, _DT] = {}
        global_indices: Dict[str, _DT] = {}

        for etype in ('node', 'cell'):
            local_bdry_flag = getattr(local_mesh, f"boundary_{etype}_flag")()
            global_flags_on_bdry[etype] = global_bdry_flags[etype][flags[etype]][local_bdry_flag]
            src_id_on_bdry[etype] = src_id[etype][flags[etype]][local_bdry_flag]
            local_id_on_bdry[etype] = local_id[etype][flags[etype]][local_bdry_flag]

            all_global_indices = bm.arange(mesh.count(etype), dtype=mesh.itype, device=mesh.device)
            global_indices[etype] = all_global_indices[flags[etype]]

        yield (local_mesh, global_flags_on_bdry, src_id_on_bdry, local_id_on_bdry, global_indices)


### Genearte Tensor data on a field, such as mesh entities.

class ParallelMesh(Generic[_MT]):
    # Required attributes
    _id : int
    _mesh : _MT
    _global_bdry_flags : Dict[str, _DT] # Bool(8)
    _src_id_on_bdry : Dict[str, _DT] # UInt8
    _virtual_table : Dict[str, _DT] # itype=Int32
    # Optional attributes: Generate if not given
    _global_indices : Dict[str, _DT] # NOTE: Is this necessary?
    # Other attributes
    _global_offsets : Dict[str, int]
    _process_table : List[int]

    class BType(IntEnum):
        """Boundary entity type"""
        GLOBAL = 1
        VIRTUAL = 2

    def __init__(self, id: int, mesh: _MT,
                 global_bdry_flags: Dict[str, _DT],
                 src_id_on_bdry: Dict[str, _DT],
                 virtual_table : Dict[str, _DT],
                 global_indices : Optional[Dict[str, _DT]] = None, *,
                 comm: Optional[Comm] = None):
        self._id = int(id)
        self._mesh = mesh
        self._global_bdry_flags = global_bdry_flags
        self._src_id_on_bdry = src_id_on_bdry
        self._virtual_table = virtual_table
        self._global_indices = global_indices

        self._comm = comm if comm else COMM_WORLD
        self._Make_process_table()

    def __getattr__(self, name):
        return getattr(self._mesh, name)

    ### MPI

    @property
    def mpi_rank(self):
        return self._comm.Get_rank()

    @property
    def mpi_size(self):
        return self._comm.Get_size()

    def get_comm(self):
        return self._comm

    def real_flag(self, etype: str, /):
        bdry_flag = getattr(self, f"boundary_{etype}_flag")()
        flag = ~bdry_flag
        bdry_indices = bm.nonzero(bdry_flag)[0]
        assets_bdry_indices = bdry_indices[~self.virtual_flag_on_boundary(etype)]
        flag = bm.set_at(flag, assets_bdry_indices, True)
        return flag

    def virtual_indices(self, etype: str, /):
        """Return indices of virtual entity in the partition."""
        bdry_flag = getattr(self, f"boundary_{etype}_flag")()
        bdry_indices = bm.nonzero(bdry_flag)[0]
        return bdry_indices[self.virtual_flag_on_boundary(etype)]

    def global_indices(self, etype: str, /):
        return self._global_indices[etype]

    def _Make_process_table(self) -> None:
        SIZE = self.mpi_size
        send_buf = [self._id,] * SIZE
        recv_buf = self._comm.alltoall(send_buf)
        self._process_table = [None,] * SIZE

        for pro_id in range(SIZE):
            par_id = recv_buf[pro_id]
            self._process_table[par_id] = pro_id

    def Converge(self, etype: str, data_on_bdry: _DT, /) -> List[Union[Tuple[_DT, _DT], None]]:
        """Every virtual entities send data to their reals.

        Return message from other partitions as a list, ordered by partition ID
        (NOT process ID). Each message is given as a 2-tuple, containing the
        gathered data and their local indices in this partition. None for self.
        """
        SIZE = self.mpi_size
        id_on_bdry = self._src_id_on_bdry[etype]
        virtual_table = self._virtual_table[etype]
        send_buf = [None,] * SIZE

        for par_id in range(SIZE):
            if par_id == self._id:
                continue
            pro_id = self._process_table[par_id]
            mask = (id_on_bdry == par_id)
            send_buf[pro_id] = (data_on_bdry[mask], virtual_table[mask])

        recv_buf = self._comm.alltoall(send_buf)
        result = []

        for par_id in range(SIZE):
            pro_id = self._process_table[par_id]
            result.append(recv_buf[pro_id])

        return result

    def Broadcast(self, etype: str, data_on_bdry: _DT, /):
        """Every virtual entities fetch data from their reals."""
        pass

    ### Counter

    def count(self, etype):
        return self._mesh.count(etype)

    def Count_all(self, etype: str, /) -> int:
        num_real = self.count_real(etype)
        return self._comm.allreduce(num_real)

    def count_real(self, etype: str, /) -> int:
        num = self.count(etype)
        num_virtual = bm.sum(self.virtual_flag_on_boundary(etype), dtype=self.itype)
        return num - num_virtual

    ### Entity

    def cell_to_global_node(self):
        pass

    ### Boundary

    def global_flag_on_boundary(self, etype: str, /):
        return self._global_bdry_flags[etype]

    def virtual_flag_on_boundary(self, etype: str, /):
        return self._src_id_on_bdry[etype] != self._id

    def real_flag_on_boundary(self, etype: str, /):
        return self._src_id_on_bdry[etype] == self._id

    def partial_boundary_index(self, etype: str, /, bdry_indices: Optional[_DT] = None):
        """Return the local indices of entities which are on the global boundary
        and owned by this partition."""
        global_bdry_flags = self.global_flag_on_boundary(etype)
        real_flag_on_bdry = self.real_flag_on_boundary(etype)
        mask = bm.logical_and(real_flag_on_bdry, global_bdry_flags)
        del global_bdry_flags, real_flag_on_bdry
        if bdry_indices is None:
            bdry_flag = getattr(self, f"boundary_{etype}_flag")()
            bdry_indices = bm.nonzero(bdry_flag)[0]
        return bdry_indices[mask]

    ### Ipoints

    def edge_to_global_ipoints(self, p: int, index: Index = _S):
        NN = self.number_of_nodes()
        NE = self.count_assets('edge')
        assets_indices = self.assets_flag('edge')
        edges = self.edge[assets_indices][index]
        kwargs = bm.context(edges)
        indices = bm.arange(NE, **kwargs)[index]
        return bm.concatenate([
            edges[:, 0].reshape(-1, 1),
            (p-1) * indices.reshape(-1, 1) + bm.arange(0, p-1, **kwargs) + NN,
            edges[:, 1].reshape(-1, 1),
        ], axis=-1)
