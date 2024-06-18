"""Neighbors search backends.

Source:
https://github.com/tumaer/lagrangebench/blob/main/lagrangebench/case_setup/partition.py
"""

from functools import partial
from typing import Optional

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import numpy as onp
from jax import jit
from jax_md.partition import (
    CellList,
    MaskFn,
    NeighborFn,
    NeighborList,
    NeighborListFns,
    NeighborListFormat,
    _displacement_or_metric_to_metric_sq,
    _neighboring_cells,
    _shift_array,
    cell_list,
    is_format_valid,
    is_sparse,
    space,
)
from jax_md.partition import neighbor_list as vmap_neighbor_list


def get_particle_cells(idx, cl_capacity, N):
    """
    Given a cell list idx of shape (nx, ny, nz, cell_capacity), we first
    enumerate each cell and then return a list of shape (N,) containing the
    number of the cell each particle belongs to.
    """
    # containes particle indices in each cell (num_cells, cell_capacity)
    idx = idx.reshape(-1, cl_capacity)

    # (num_cells, cell_capacity) of
    # [[0,0,...0],[1,1,...1],...,[num_cells-1,num_cells-1,...num_cells-1]
    list_cells = jnp.broadcast_to(jnp.arange(idx.shape[0])[:, None], idx.shape)

    idx = jnp.reshape(idx, (-1,))  # flatten
    list_cells = jnp.reshape(list_cells, (-1,))  # flatten

    ordering = jnp.argsort(idx)  # each particle is only once in the cell list
    particle_cells = list_cells[ordering][:N]
    return particle_cells


def _scan_neighbor_list(
    displacement_or_metric: space.DisplacementOrMetricFn,
    box_size: space.Box,
    r_cutoff: float,
    dr_threshold: float = 0.0,
    capacity_multiplier: float = 1.25,
    disable_cell_list: bool = False,
    mask_self: bool = True,
    custom_mask_function: Optional[MaskFn] = None,
    fractional_coordinates: bool = False,
    format: NeighborListFormat = NeighborListFormat.Sparse,
    num_partitions: int = 1,
    **static_kwargs,
) -> NeighborFn:
    """Modified JAX-MD neighbor list function that uses `lax.scan` to compute the
    distance between particles to save memory.

    Original: https://github.com/jax-md/jax-md/blob/main/jax_md/partition.py

    Returns a function that builds a list neighbors for collections of points.

    Neighbor lists must balance the need to be jit compatible with the fact that
    under a jit the maximum number of neighbors cannot change (owing to static
    shape requirements). To deal with this, our `neighbor_list` returns a
    `NeighborListFns` object that contains two functions: 1)
    `neighbor_fn.allocate` create a new neighbor list and 2) `neighbor_fn.update`
    updates an existing neighbor list. Neighbor lists themselves additionally
    have a convenience `update` member function.

    Note that allocation of a new neighbor list cannot be jit compiled since it
    uses the positions to infer the maximum number of neighbors (along with
    additional space specified by the `capacity_multiplier`). Updating the
    neighbor list can be jit compiled; if the neighbor list capacity is not
    sufficient to store all the neighbors, the `did_buffer_overflow` bit
    will be set to `True` and a new neighbor list will need to be reallocated.

    Here is a typical example of a simulation loop with neighbor lists:

    .. code-block:: python

        init_fn, apply_fn = simulate.nve(energy_fn, shift, 1e-3)
        exact_init_fn, exact_apply_fn = simulate.nve(exact_energy_fn, shift, 1e-3)

        nbrs = neighbor_fn.allocate(R)
        state = init_fn(random.PRNGKey(0), R, neighbor_idx=nbrs.idx)

        def body_fn(i, state):
        state, nbrs = state
        nbrs = nbrs.update(state.position)
        state = apply_fn(state, neighbor_idx=nbrs.idx)
        return state, nbrs

        step = 0
        for _ in range(20):
        new_state, nbrs = lax.fori_loop(0, 100, body_fn, (state, nbrs))
        if nbrs.did_buffer_overflow:
            nbrs = neighbor_fn.allocate(state.position)
        else:
            state = new_state
            step += 1

    Args:
        displacement: A function `d(R_a, R_b)` that computes the displacement
        between pairs of points.
        box_size: Either a float specifying the size of the box or an array of
        shape `[spatial_dim]` specifying the box size in each spatial dimension.
        r_cutoff: A scalar specifying the neighborhood radius.
        dr_threshold: A scalar specifying the maximum distance particles can move
        before rebuilding the neighbor list.
        capacity_multiplier: A floating point scalar specifying the fractional
        increase in maximum neighborhood occupancy we allocate compared with the
        maximum in the example positions.
        disable_cell_list: An optional boolean. If set to `True` then the neighbor
        list is constructed using only distances. This can be useful for
        debugging but should generally be left as `False`.
        mask_self: An optional boolean. Determines whether points can consider
        themselves to be their own neighbors.
        custom_mask_function: An optional function. Takes the neighbor array
        and masks selected elements. Note: The input array to the function is
        `(n_particles, m)` where the index of particle 1 is in index in the first
        dimension of the array, the index of particle 2 is given by the value in
        the array
        fractional_coordinates: An optional boolean. Specifies whether positions
        will be supplied in fractional coordinates in the unit cube, :math:`[0, 1]^d`.
        If this is set to True then the `box_size` will be set to `1.0` and the
        cell size used in the cell list will be set to `cutoff / box_size`.
        format: The format of the neighbor list; see the :meth:`NeighborListFormat` enum
        for details about the different choices for formats. Defaults to `Dense`.
        **static_kwargs: kwargs that get threaded through the calculation of
        example positions.
    Returns:
        A NeighborListFns object that contains a method to allocate a new neighbor
        list and a method to update an existing neighbor list.
    """

    assert disable_cell_list is False, "Works only with a cell list"
    assert format == NeighborListFormat.Sparse, "Works only with sparse neighbor list"
    assert custom_mask_function is None, "Custom masking not implemented"
    # assert mask_self == False, "Self edges cannot be excluded for now"

    is_format_valid(format)
    box_size = lax.stop_gradient(box_size)
    r_cutoff = lax.stop_gradient(r_cutoff)
    dr_threshold = lax.stop_gradient(dr_threshold)

    box_size = jnp.float32(box_size)

    cutoff = r_cutoff + dr_threshold
    cutoff_sq = cutoff**2
    threshold_sq = (dr_threshold / jnp.float32(2)) ** 2
    metric_sq = _displacement_or_metric_to_metric_sq(displacement_or_metric)

    cell_size = cutoff
    if fractional_coordinates:
        cell_size = cutoff / box_size
        box_size = (
            jnp.float32(box_size)
            if onp.isscalar(box_size)
            else onp.ones_like(box_size, jnp.float32)
        )

    assert jnp.all(cell_size < box_size / 3.0), "Don't use scan with very few cells"

    cl_fn = cell_list(box_size, cell_size, capacity_multiplier)

    @jit
    def cell_list_candidate_fn(cl: CellList, position: jnp.ndarray) -> jnp.ndarray:
        N, dim = position.shape

        idx = cl.id_buffer

        cell_idx = [idx]

        for dindex in _neighboring_cells(
            dim
        ):  # here the expansion happens over all adjacent cells happens
            if onp.all(dindex == 0):
                continue
            cell_idx += [_shift_array(idx, dindex)]  # 27* (nx,ny,nz,cell_capacity, 1)

        cell_idx = jnp.concatenate(cell_idx, axis=-2)
        cell_idx = cell_idx[..., jnp.newaxis, :, :]  # (nx,ny,nz,1,27*cell_capacity, 1)
        cell_idx = jnp.broadcast_to(
            cell_idx, idx.shape[:-1] + cell_idx.shape[-2:]
        )  # (nx,ny,nz,cell_capacity,27*cell_capacity) TODO: memory blows up here

        def copy_values_from_cell(value, cell_value, cell_id):
            scatter_indices = jnp.reshape(cell_id, (-1,))  # (nx*ny*nz*cell_capacity)
            cell_value = jnp.reshape(
                cell_value, (-1,) + cell_value.shape[-2:]
            )  # (nx*ny*nz*cell_capacity, 27*cell_capacity, 1)
            return value.at[scatter_indices].set(cell_value)

        neighbor_idx = jnp.zeros(
            (N + 1,) + cell_idx.shape[-2:], jnp.int32
        )  # (N, 27*cell_capacity, 1) TODO: too much memory
        neighbor_idx = copy_values_from_cell(neighbor_idx, cell_idx, idx)
        return neighbor_idx[:-1, :, 0]  # shape (N, 27*cell_capacity)

    @jit
    def prune_neighbor_list_sparse(
        position: jnp.ndarray, idx: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        d = partial(metric_sq, **kwargs)
        d = space.map_bond(d)

        N = position.shape[0]
        sender_idx = jnp.broadcast_to(jnp.arange(N)[:, None], idx.shape)

        sender_idx = jnp.reshape(
            sender_idx, (-1,)
        )  # (N, 27*cell_capacity) -> (N*27*cell_capacity)
        receiver_idx = jnp.reshape(idx, (-1,))
        dR = d(
            position[sender_idx], position[receiver_idx]
        )  # (N*27*cell_capacity) eventually 3x during computation

        mask = (dR < cutoff_sq) & (receiver_idx < N)
        if format is NeighborListFormat.OrderedSparse:
            mask = mask & (receiver_idx < sender_idx)

        out_idx = N * jnp.ones(receiver_idx.shape, jnp.int32)

        cumsum = jnp.cumsum(mask)
        index = jnp.where(
            mask, cumsum - 1, len(receiver_idx) - 1
        )  # 7th object of shape (N*27*cell_capacity)
        receiver_idx = out_idx.at[index].set(receiver_idx)
        sender_idx = out_idx.at[index].set(sender_idx)
        max_occupancy = cumsum[-1]

        return jnp.stack((receiver_idx, sender_idx)), max_occupancy

    def neighbor_list_fn(
        position: jnp.ndarray,
        neighbors: Optional[NeighborList] = None,
        extra_capacity: int = 0,
        **kwargs,
    ) -> NeighborList:
        nbrs = neighbors

        def neighbor_fn(position_and_overflow, max_occupancy=None):
            position, overflow = position_and_overflow
            N = position.shape[0]

            if neighbors is None:  # cl.shape = (nx, ny, nz, cell_capacity, dim)
                cl = cl_fn.allocate(position, extra_capacity=extra_capacity)
            else:
                cl = cl_fn.update(position, neighbors.cell_list_capacity)
            overflow = overflow | cl.did_buffer_overflow
            cl_capacity = cl.cell_capacity

            if num_partitions == 1:
                implementation = "original"
            elif num_partitions > 1:
                implementation = (
                    "numcells"  # "numcells", "twentyseven", "vanilla", "original"
                )

            if implementation == "numcells":
                # idx = cell_list_candidate_fn(cl, position)
                #   # idx.shape = (N, 27*cell_capacity)
                # print("82 ", get_gpu_stats())
                # idx, occupancy = prune_neighbor_list_sparse(position, idx, **kwargs)
                ################################################################

                N, dim = position.shape

                idx = cl.id_buffer

                cell_idx = [idx]

                for dindex in _neighboring_cells(
                    dim
                ):  # here the expansion happens over all adjacent cells happens
                    if onp.all(dindex == 0):
                        continue
                    cell_idx += [
                        _shift_array(idx, dindex)
                    ]  # 27* (nx,ny,nz,cell_capacity, 1)

                cell_idx = jnp.concatenate(cell_idx, axis=-2)
                cell_idx = jnp.reshape(
                    cell_idx, (-1, cell_idx.shape[-2])
                )  # (num_cells, num_potential_connections)
                num_cells, considered_neighbors = cell_idx.shape

                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Given is a cell list `cell_idx` of shape (nx, ny, nz, cell_capacity).
                # Find which cell indices correspond to particle 0, 1, 2, ..., N-1
                # and write the results into a new array of shape (N, nx, ny, nz)

                def scan_body(carry, input):
                    occupancy = carry
                    slice_from = input

                    _entries = lax.dynamic_slice(
                        particle_cells, (slice_from,), (N_sub,)
                    )
                    _idx = cell_idx[_entries]

                    if mask_self:
                        particle_idx = slice_from + jnp.arange(N_sub)
                        _idx = jnp.where(_idx == particle_idx[:, None], N, _idx)

                    if num_pad > 0:
                        _idx = jnp.where(_entries[:, None] != -1, _idx, N)

                    sender_idx = (
                        jnp.broadcast_to(
                            jnp.arange(N_sub, dtype="int32")[:, None], _idx.shape
                        )
                        + slice_from
                    )
                    if num_pad > 0:
                        sender_idx = jnp.clip(sender_idx, a_max=N)

                    sender_idx = jnp.reshape(
                        sender_idx, (-1,)
                    )  # (N, 27*cell_capacity) -> (N*27*cell_capacity)
                    receiver_idx = jnp.reshape(_idx, (-1,))
                    dR = d(
                        position[sender_idx], position[receiver_idx]
                    )  # (N*27*cell_capacity) eventually 3x during computation

                    mask = (dR < cutoff_sq) & (receiver_idx < N)
                    out_idx = N * jnp.ones(receiver_idx.shape, jnp.int32)

                    cumsum = jnp.cumsum(mask)  # + occupancy
                    index = jnp.where(
                        mask, cumsum - 1, considered_neighbors * N - 1
                    )  # (N*27*cell_capacity)
                    receiver_idx = out_idx.at[index].set(receiver_idx)
                    sender_idx = out_idx.at[index].set(sender_idx)
                    occupancy += cumsum[-1]

                    carry = occupancy
                    y = jnp.stack(
                        (receiver_idx[:num_edges_sub], sender_idx[:num_edges_sub])
                    )
                    overflow = cumsum[-1] > num_edges_sub
                    return carry, (y, overflow)

                particle_cells = get_particle_cells(idx, cl_capacity, N)

                d = partial(metric_sq, **kwargs)
                d = space.map_bond(d)

                N_sub = int(
                    np.ceil(N / num_partitions)
                )  # to pad the last chunk with < num_partitions entries
                num_pad = N_sub * num_partitions - N
                particle_cells = jnp.pad(
                    particle_cells,
                    (
                        0,
                        num_pad,
                    ),
                    constant_values=-1,
                )

                if dim == 2:
                    # area of a circle with r=1/3 is 0.15514 of a unit cube volume
                    volumetric_factor = 0.34907
                elif dim == 3:
                    # volume of sphere with r=1/3 is 0.15514 of a unit cube volume
                    volumetric_factor = 0.15514

                num_edges_sub = int(
                    N_sub
                    * considered_neighbors
                    * volumetric_factor
                    * capacity_multiplier
                )

                carry = jnp.array(0)
                xs = jnp.array([i * N_sub for i in range(num_partitions)])
                # print("82 (numcells)", get_gpu_stats())
                occupancy, (idx, overflows) = lax.scan(
                    scan_body, carry, xs, length=num_partitions
                )
                # print("83 ", get_gpu_stats())
                overflow = overflow | overflows.sum()

                # print(f"idx memory: {idx.nbytes / 1e6:.0f}MB, idx.shape={idx.shape},
                #   cl.id_buffer.shape={cl.id_buffer.shape}" )
                idx = idx.transpose(1, 2, 0).reshape(2, -1)

                # sort to enable pruning later
                ordering = jnp.argsort(idx[1])
                idx = idx[:, ordering]

                if max_occupancy is None:
                    _extra_capacity = N * extra_capacity
                    max_occupancy = int(
                        occupancy * capacity_multiplier + _extra_capacity
                    )
                    if max_occupancy > idx.shape[-1]:
                        max_occupancy = idx.shape[-1]
                    if not is_sparse(format):
                        capacity_limit = N - 1 if mask_self else N
                    elif format is NeighborListFormat.Sparse:
                        capacity_limit = N * (N - 1) if mask_self else N**2
                    else:
                        capacity_limit = N * (N - 1) // 2
                    if max_occupancy > capacity_limit:
                        max_occupancy = capacity_limit

                # prune neighbors list to max_occupancy by removing paddings
                idx = idx[:, :max_occupancy]
            elif implementation == "original_expanded":
                # TODO: here we expand on the 27 adjacent cells
                ####################################################################
                ###
                # idx = cell_list_candidate_fn(cl, position)
                #   # shape (N, 27*cell_capacity) -> 19M too much!
                N, dim = position.shape

                idx = cl.id_buffer  # (5, 5, 5, 88, 1)

                cell_idx = [idx]

                for dindex in _neighboring_cells(
                    dim
                ):  # here the expansion happens over all adjacent cells happens
                    if onp.all(dindex == 0):
                        continue
                    cell_idx += [
                        _shift_array(idx, dindex)
                    ]  # 27* (nx,ny,nz,cell_capacity)

                cell_idx = jnp.concatenate(cell_idx, axis=-2)  # (5, 5, 5, 2376, 1)
                cell_idx = cell_idx[..., jnp.newaxis, :, :]  # (5, 5, 5, 1, 2376, 1)
                # TODO: memory blows up here by factor "cell_capacity"
                cell_idx = jnp.broadcast_to(
                    cell_idx, idx.shape[:-1] + cell_idx.shape[-2:]
                )  # 1.2*X (nx,ny,nz,cell_capacity,27*cell_capacity)

                # def copy_values_from_cell(value, cell_value, cell_id):
                #     scatter_indices = jnp.reshape(cell_id, (-1,))
                #     cell_value = jnp.reshape(cell_value, (-1,) +cell_value.shape[-2:])
                #     return value.at[scatter_indices].set(cell_value)
                # TODO: further memory increase in the next two lines
                neighbor_idx = jnp.zeros(
                    (N + 1,) + cell_idx.shape[-2:], jnp.int32
                )  # X (N, 27*cell_capacity, 1)
                # neighbor_idx = copy_values_from_cell(neighbor_idx, cell_idx, idx) # X
                scatter_indices = jnp.reshape(
                    idx, (-1,)
                )  # (11000,) each cell allocation over all cells expanded
                cell_value = jnp.reshape(
                    cell_idx, (-1,) + cell_idx.shape[-2:]
                )  # (nx*ny*nz*cell_capacity, 27*cell_capacity, 1)
                neighbor_idx = neighbor_idx.at[scatter_indices].set(
                    cell_value
                )  # X (N, 27*cell_capacity, 1)

                idx = neighbor_idx[
                    :-1, :, 0
                ]  # X shape (N, 27*cell_capacity) this only removes the 8001th element
                # this is just expanded over all cells indices. Should work with
                #   arbitrary pices over the last dimension

                ####################################################################
                # idx.shape = (nx*ny*nz*cell_capacity**2*27)
                #   -> 26M (or actually just 19M) too much!
                # idx, occupancy = prune_neighbor_list_sparse(position, idx, **kwargs)
                d = partial(metric_sq, **kwargs)
                d = space.map_bond(d)

                N = position.shape[0]
                sender_idx = jnp.broadcast_to(
                    jnp.arange(N)[:, None], idx.shape
                )  # 2X (N, 27*cell_capacity)

                sender_idx = jnp.reshape(
                    sender_idx, (-1,)
                )  # (N, 27*cell_capacity) -> (N*27*cell_capacity)
                # [0,0,0,0...0, 1,1,1,1...1, ....]
                receiver_idx = jnp.reshape(
                    idx, (-1,)
                )  # flatten the stuff with all possible neighbors (27*cell_size) of
                # particle 0, of 1, ....
                dR = d(
                    position[sender_idx], position[receiver_idx]
                )  # (N*27*cell_capacity) eventually 3x during computation

                mask = (dR < cutoff_sq) & (receiver_idx < N)  # negligible
                # if format is NeighborListFormat.OrderedSparse:
                #     mask = mask & (receiver_idx < sender_idx)

                out_idx = N * jnp.ones(receiver_idx.shape, jnp.int32)  # X
                cumsum = jnp.cumsum(mask)  # 2X
                index = jnp.where(
                    mask, cumsum - 1, len(receiver_idx) - 1
                )  # 2X 7th object of shape (N*27*cell_capacity)
                receiver_idx = out_idx.at[index].set(
                    receiver_idx
                )  # X # this operation sorts the entries
                sender_idx = out_idx.at[index].set(
                    sender_idx
                )  # 2X -> X # this operation also sorts the entries
                max_occupancy_ = cumsum[-1]

                idx, occupancy = jnp.stack((receiver_idx, sender_idx)), max_occupancy_
                # Memory: idx 2X, neihbor_idx X, cell_idx 0.8X, sender_idx X,
                #   receiver_idx X, dR 2X, out_idx X, cumsum 2X, index 2X
                # idx_final = jnp.zeros((N, max_occupancy), jnp.int32) # X -> 2X
                # print("max occupancy2 ", occupancy)

                if max_occupancy is None:
                    _extra_capacity = N * extra_capacity
                    max_occupancy = int(
                        occupancy * capacity_multiplier + _extra_capacity
                    )
                    if max_occupancy > idx.shape[-1]:
                        max_occupancy = idx.shape[-1]
                    if not is_sparse(format):
                        capacity_limit = N - 1 if mask_self else N
                    elif format is NeighborListFormat.Sparse:
                        capacity_limit = N * (N - 1) if mask_self else N**2
                    else:
                        capacity_limit = N * (N - 1) // 2
                    if max_occupancy > capacity_limit:
                        max_occupancy = capacity_limit
                idx = idx[
                    :, :max_occupancy
                ]  # shape (N, max_occupancy) -> 2M much smaller
                # TODO: from here on the size is ~10x smaller after
                #   idx=idx[:, :max_occupancy]
                # how can we run the previous part sequentially?
                ###
                ####################################################################
            elif implementation == "original":
                # print("82 (original)", get_gpu_stats())
                idx = cell_list_candidate_fn(cl, position)

                idx, occupancy = prune_neighbor_list_sparse(position, idx, **kwargs)

                if max_occupancy is None:
                    _extra_capacity = (
                        extra_capacity if not is_sparse(format) else N * extra_capacity
                    )
                    max_occupancy = int(
                        occupancy * capacity_multiplier + _extra_capacity
                    )
                    if max_occupancy > idx.shape[-1]:
                        max_occupancy = idx.shape[-1]
                    if not is_sparse(format):
                        capacity_limit = N - 1 if mask_self else N
                    elif format is NeighborListFormat.Sparse:
                        capacity_limit = N * (N - 1) if mask_self else N**2
                    else:
                        capacity_limit = N * (N - 1) // 2
                    if max_occupancy > capacity_limit:
                        max_occupancy = capacity_limit
                idx = idx[:, :max_occupancy]

                # print("83 ", get_gpu_stats())
                # print(f"idx memory: {idx.nbytes / 1e6:.0f}MB, idx.shape={idx.shape},
                #   cl.id_buffer.shape={cl.id_buffer.shape}" )

            # print("##### max occupancy", max_occupancy, "occupancy", occupancy)

            update_fn = neighbor_list_fn if neighbors is None else neighbors.update_fn
            return NeighborList(
                idx,
                position,
                overflow | (occupancy > max_occupancy),
                cl_capacity,
                max_occupancy,
                format,
                update_fn,
            )  # pytype: disable=wrong-arg-count

        if nbrs is None:
            return neighbor_fn((position, False))

        neighbor_fn = partial(neighbor_fn, max_occupancy=nbrs.max_occupancy)

        d = partial(metric_sq, **kwargs)
        d = jax.vmap(d)

        return lax.cond(
            jnp.any(d(position, nbrs.reference_position) > threshold_sq),
            (position, nbrs.did_buffer_overflow),
            neighbor_fn,
            nbrs,
            lambda x: x,
        )

    def allocate_fn(
        position: jnp.ndarray, extra_capacity: int = 0, **kwargs
    ) -> NeighborList:
        return neighbor_list_fn(position, extra_capacity=extra_capacity, **kwargs)

    @jit
    def update_fn(
        position: jnp.ndarray, neighbors: NeighborList, **kwargs
    ) -> NeighborList:
        return neighbor_list_fn(position, neighbors, **kwargs)

    return NeighborListFns(allocate_fn, update_fn)  # pytype: disable=wrong-arg-count


def _matscipy_neighbor_list(
    displacement_or_metric: space.DisplacementOrMetricFn,
    box_size: space.Box,
    r_cutoff: float,
    dr_threshold: float = 0.0,
    capacity_multiplier: float = 1.25,
    disable_cell_list: bool = False,
    mask_self: bool = True,
    custom_mask_function: Optional[MaskFn] = None,
    fractional_coordinates: bool = False,
    format: NeighborListFormat = NeighborListFormat.Dense,
    **static_kwargs,
) -> NeighborFn:
    pbc = static_kwargs["pbc"]
    num_particles_max = static_kwargs["num_particles_max"]

    from matscipy.neighbours import neighbour_list as matscipy_nl

    assert box_size.ndim == 1 and (len(box_size) in [2, 3])
    if box_size.shape == (2,):
        box_size = np.pad(box_size, (0, 1), mode="constant", constant_values=1.0)
    if box_size.shape != (3, 3):
        box_size = np.diag(box_size)

    if len(pbc) == 2:
        pbc = np.pad(pbc, (0, 1), mode="constant", constant_values=False)
    else:
        pbc = np.asarray(pbc, dtype=bool)

    def matscipy_wrapper(position, idx_shape, num_particles):
        position = position[:num_particles]
        if position.shape[1] == 2:
            position = np.pad(
                position, ((0, 0), (0, 1)), mode="constant", constant_values=0.5
            )
        edge_list = matscipy_nl(
            "ij", cutoff=r_cutoff, positions=position, cell=box_size, pbc=pbc
        )
        edge_list = np.asarray(edge_list, dtype=np.int32)
        if not mask_self:
            # add self connection, which matscipy does not do
            self_connect = np.arange(num_particles, dtype=np.int32)
            self_connect = np.array([self_connect, self_connect])
            edge_list = np.concatenate((self_connect, edge_list), axis=-1)

        if edge_list.shape[1] > idx_shape[1]:  # overflow true case
            idx_new = np.asarray(edge_list[:, : idx_shape[1]])
            buffer_overflow = np.array(True)
        else:
            idx_new = np.ones(idx_shape, dtype=np.int32) * num_particles_max
            idx_new[:, : edge_list.shape[1]] = edge_list
            buffer_overflow = np.array(False)

        return idx_new, buffer_overflow

    @jax.jit
    def update_fn(
        position: jnp.ndarray, neighbors: NeighborList, **kwargs
    ) -> NeighborList:
        num_particles = kwargs["num_particles"]

        shape_edgelist = jax.ShapeDtypeStruct(
            neighbors.idx.shape, dtype=neighbors.idx.dtype
        )
        shape_overflow = jax.ShapeDtypeStruct((), dtype=bool)
        shape_out = (shape_edgelist, shape_overflow)
        idx, buffer_overflow = jax.pure_callback(
            matscipy_wrapper, shape_out, position, neighbors.idx.shape, num_particles
        )
        return NeighborList(
            idx,
            position,
            buffer_overflow,
            None,
            None,
            None,
            update_fn,
        )

    def allocate_fn(
        position: jnp.ndarray, extra_capacity: int = 0, **kwargs
    ) -> NeighborList:
        num_particles = kwargs["num_particles"]

        position = position[:num_particles]

        if position.shape[1] == 2:
            position = np.pad(
                position, ((0, 0), (0, 1)), mode="constant", constant_values=0.5
            )

        edge_list = matscipy_nl(
            "ij", cutoff=r_cutoff, positions=position, cell=box_size, pbc=pbc
        )
        edge_list = np.asarray(edge_list, dtype=np.int32)
        if not mask_self:
            # add self connection, which matscipy does not do
            self_connect = np.arange(num_particles, dtype=np.int32)
            self_connect = np.array([self_connect, self_connect])
            edge_list = np.concatenate((self_connect, edge_list), axis=-1)

        # in case this is a (2,M) pair list, we pad with N and capacity_multiplier
        factor = capacity_multiplier * num_particles_max / num_particles
        res = num_particles * jnp.ones(
            (2, round(edge_list.shape[1] * factor + extra_capacity)),
            np.int32,
        )
        res = res.at[:, : edge_list.shape[1]].set(edge_list)
        return NeighborList(
            res,
            position,
            jnp.array(False),
            None,
            None,
            None,
            update_fn,
        )

    return NeighborListFns(allocate_fn, update_fn)


BACKENDS = {
    "jaxmd_vmap": vmap_neighbor_list,
    "jaxmd_scan": _scan_neighbor_list,
    "matscipy": _matscipy_neighbor_list,
}


def neighbor_list(
    displacement_or_metric: space.DisplacementOrMetricFn,
    box_size: space.Box,
    r_cutoff: float,
    backend: str = "jaxmd_vmap",
    dr_threshold: float = 0.0,
    capacity_multiplier: float = 1.25,
    disable_cell_list: bool = False,
    mask_self: bool = True,
    custom_mask_function: Optional[MaskFn] = None,
    fractional_coordinates: bool = False,
    format: NeighborListFormat = NeighborListFormat.Sparse,
    num_particles_max: int = None,
    num_partitions: int = 1,
    pbc: jnp.ndarray = None,
) -> NeighborFn:
    """Neighbor lists wrapper.

    Args:
        backend: The backend to use. One of "jaxmd_vmap", "jaxmd_scan", "matscipy".

            - "jaxmd_vmap": Default jax-md neighbor list. Uses vmap. Fast.
            - "jaxmd_scan": Modified jax-md neighbor list. Uses scan. Memory efficient.
            - "matscipy": Matscipy neighbor list. Runs on cpu, allows dynamic shapes.
    """
    assert backend in BACKENDS, f"Unknown backend {backend}"

    return BACKENDS[backend](
        displacement_or_metric,
        box_size,
        r_cutoff,
        dr_threshold,
        capacity_multiplier,
        disable_cell_list,
        mask_self,
        custom_mask_function,
        fractional_coordinates,
        format,
        num_particles_max=num_particles_max,
        num_partitions=num_partitions,
        pbc=pbc,
    )


if __name__ == "__main__":
    # edge_list = matscipy_nl(
    #     "ij",
    #     cutoff=0.45,
    #     positions=np.array([[0.1, 0.1, 0.1], [0.5, 0.1, 0.1], [0.5, 0.5, 0.1]]),
    #     cell=np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    #     pbc=np.array([True, True, True])
    # )

     box_size = np.array([1.0, 1.0])
     r = np.array([[0.1, 0.1], [0.5, 0.1]])
     displacement_fn, shift_fn = space.periodic(side=box_size)
     r_cutoff = 0.45

     neighbor_fn = neighbor_list(
         displacement_fn,
         box_size,
         r_cutoff=r_cutoff,
         backend="jaxmd_vmap",
         dr_threshold=r_cutoff * 0.25,
         capacity_multiplier=1.25,
         mask_self=False,
     )
     neighbors = neighbor_fn.allocate(r)
     neighbors = neighbors.update(r)
     neighbors = neighbors.update(r)
    # a = np.array([1,2,3])
    # f = lambda x: x**2

    # def distance_fn(x, y):
    #     return lax.scan(lambda _, x: (None, d(*x)), None, (x, y))[1]

    # def scan(f, init, xs, length=None):
    #     if xs is None:
    #         xs = [None] * length
    #     carry = init
    #     ys = []
    #     for x in xs:
    #         carry, y = f(carry, x)
    #         ys.append(y)
    #     return carry, np.stack(ys)

