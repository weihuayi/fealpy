from __future__ import annotations

__all__ = [
    "EntityMPI",
    "dist_from_masks",
    "mapped_masks",
]

from typing import NamedTuple, TypeVar
from collections.abc import Sequence
from mpi4py.MPI import Comm, COMM_WORLD

from ..backend import bm, Tensor


_T = TypeVar("_T")


class SharingPair(NamedTuple):
    index_self: Tensor
    index_other: Tensor


class SparseData1D(NamedTuple):
    data: Tensor
    indices: Tensor


class EntityMPI:
    """The *Entity Message Passing Interface* (EMPI) in FEALPy"""
    _id: int
    _process_map: list[int]
    _global_indices: Tensor | None
    _sharing_pairs: list[SharingPair | None]

    def __init__(
        self,
        indices: Tensor | None = None,
        pairs: list[SharingPair | None] = [],
        id: int | None = None,
        *,
        comm: Comm | None = None
    )-> None:
        self._global_indices = indices
        self._sharing_pairs = pairs
        self._comm = comm if comm else COMM_WORLD

        if id is None:
            id = self._comm.Get_rank()
            self._process_map = list(range(self._comm.Get_size()))
        else:
            self._id = int(id)
            self._process_map = self._make_process_table()

    @property
    def mpi_rank(self):
        return self._comm.Get_rank()

    @property
    def mpi_size(self):
        return self._comm.Get_size()

    def _make_process_table(self) -> list[int]:
        SIZE = self.mpi_size
        send_buf = [self._id,] * SIZE
        recv_buf = self._comm.alltoall(send_buf)
        part2process = [0,] * SIZE

        for pro_id in range(SIZE):
            par_id = recv_buf[pro_id]
            part2process[par_id] = pro_id

        return part2process

    def _alltoall(self, data: Sequence[_T], /) -> list[_T]:
        """Exchange data between partitions."""
        SIZE = self.mpi_size
        send_buf = [
            data[self._process_map[par_id]]
            for par_id in range(SIZE)
        ]

        recv_buf = self._comm.alltoall(send_buf)

        return [
            recv_buf[self._process_map[par_id]]
            for par_id in range(SIZE)
        ]

    def _gather(self, data: _T, /, root: int = 0) -> list[_T] | None:
        recv_buf = self._comm.gather(data, root=root)
        if self._comm.Get_rank() == root:
            assert recv_buf is not None
            SIZE = self.mpi_size
            return [
                recv_buf[self._process_map[par_id]]
                for par_id in range(SIZE)
            ]
        return None

    def _scatter(self, data: list[_T] | None, /, root: int = 0) -> _T:
        if data is None:
            send_buf = None
        else:
            send_buf = [data[self._process_map[par_id]]
                        for par_id in range(self.mpi_size)]
        return self._comm.scatter(send_buf, root=root)

    def refs(self, size: int | None = None) -> Tensor:
        """Return the reference count of this entity in each partition."""
        if size is None:
            if self._global_indices is None:
                raise ValueError("global indices are required to count references "
                                 "without size.")
            size = self._global_indices.shape[0]
        count = bm.ones((size,), dtype=bm.int32)
        for pair in self._sharing_pairs:
            if pair is None:
                continue
            count[pair.index_self] += 1
        return count

    # --------------------------
    # Synchronize between partitions
    # --------------------------

    def sync(self, array: Tensor, /) -> list[SparseData1D | None]:
        """Synchronize arrays from shared partitions.

        Parameters:
            array (Tensor): data defined on shared entities.

        Returns:
            list: data from other partitions as a list, ordered by partition
            ID (NOT process ID). Each element is given as a 2-tuple, containing
            - Tensor: the received data,
            - Tensor: local indices in this partition,

            or `None` for self, as out of shared entities.
        """
        if not self._sharing_pairs:
            raise ValueError("sharing pairs are required to exchange data.")

        data: list[SparseData1D | None] = []

        for pair in self._sharing_pairs:
            if pair is None:
                data.append(None)
                continue
            data.append(
                SparseData1D(
                    bm.asarray(array[pair.index_self], copy=True),
                    pair.index_other
                )
            )

        return self._alltoall(data)

    def sync_add(self, array: Tensor, /) -> Tensor:
        """Synchronize arrays from shared partitions and add them together
        according to their sharing indices."""
        result = bm.asarray(array, copy=True)
        data_list = self.sync(array)

        for data in data_list:
            if data is None:
                continue
            result = bm.index_add(result, data.indices, data.data)

        return result

    # --------------------------
    # Gather to global
    # --------------------------

    def gather(self, array: Tensor, /, root: int = 0) -> list[SparseData1D] | None:
        """Gather arrays from each partition.

        Args:
            array (Tensor): data defined on shared entities.

        Returns:
            list: data from other partitions as a list, ordered by partition
            ID (NOT process ID). Each element is given as a 2-tuple, containing
            - Tensor: the received data,
            - Tensor: global indices,

            in root. While `None` for other partitions.
        """
        if self._global_indices is None:
            raise ValueError("global indices are required to gather data.")

        data = SparseData1D(array, self._global_indices)
        return self._gather(data, root=root)

    def gather_add(self, array: Tensor, /, root: int = 0, out: Tensor | None = None) -> Tensor | None:
        """Gather arrays from all partitions and add them to the global
        according to their global indices."""
        data_list = self.gather(array, root=root)

        if data_list is None: # root
            return None

        if out is None:
            max_size = max(bm.max(data.indices) for data in data_list) + 1 # type: ignore
            result = bm.zeros(max_size, dtype=array.dtype)
        else:
            result = out

        for data in data_list:
            result = bm.index_add(result, data.indices, data.data)

        return result

    # --------------------------
    # Scatter from global
    # --------------------------

    def bcast(self, array: Tensor | None, /, root: int = 0) -> Tensor:
        """Broadcast an array with its copy view at the global indices of each
        partition."""
        if self._global_indices is None:
            raise ValueError("global indices are required to scatter data.")

        global_indices_list = self._gather(self._global_indices, root=root)

        if global_indices_list is not None:
            assert array is not None, "array is required to scatter data in root."
            data_list = [bm.asarray(array[index], copy=True) for index in global_indices_list]
        else:
            data_list = None

        return self._scatter(data_list, root=root)


def dist_from_masks(
    masks: Sequence[Tensor],
    mapping: Tensor | None = None,
    *,
    comm: Comm | None = None
) -> EntityMPI:
    """Create a EntityMPI from a list of entity masks.

    Parameters:
        masks (Sequence of Tensor): The mask of each partition.
        mapping (Tensor, optional): mapping from masks to entities to split.
            Defaults to identity.

    Returns:
        EntityMPI: The created ParallelEntity in the current process.
    """
    comm = comm if comm else COMM_WORLD
    RANK = comm.Get_rank()
    thismask = masks[RANK]
    indices = bm.nonzero(thismask)[0]
    pairs: list[SharingPair | None] = []

    if mapping is not None:
        masks = mapped_masks(masks=masks, mapping=mapping)

    for i, mask in enumerate(masks):
        if i == RANK:
            pairs.append(None)
            continue
        pairs.append(
            SharingPair(
                bm.nonzero(mask[thismask])[0],
                bm.nonzero(thismask[mask])[0],
            )
        )

    return EntityMPI(indices, pairs, comm=comm)


def mapped_masks(
    masks: Sequence[Tensor],
    mapping: Tensor,
):
    """Map masks from one entity to another according to the given mapping."""
    num_entity = int(bm.max(mapping)) + 1 # type: ignore
    mapped_masks: list[Tensor] = []

    for mask in masks:
        mapped = bm.zeros((num_entity,), dtype=mask.dtype, device=mask.device)
        mapped[mapping[mask]] = True # type: ignore
        mapped_masks.append(mapped)

    return mapped_masks
