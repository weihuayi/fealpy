"""Neighbors search backends."""

from functools import partial
from typing import Optional

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import numpy as onp
from jax import jit

from .jax_md import space
from .jax_md.partition import (
    MaskFn,
    NeighborFn,
    NeighborList,
    NeighborListFns,
    NeighborListFormat,
    PartitionError,
    PartitionErrorCode,
    _displacement_or_metric_to_metric_sq,
    _neighboring_cells,
    cell_list,
    is_format_valid,
    is_sparse,
    shift_array,
)
from .jax_md.partition import neighbor_list as vmap_neighbor_list

PEC = PartitionErrorCode


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
    box: space.Box,
    r_cutoff: float,
    dr_threshold: float = 0.0,
    capacity_multiplier: float = 1.25,
    disable_cell_list: bool = False,
    mask_self: bool = True,
    custom_mask_function: Optional[MaskFn] = None,
    fractional_coordinates: bool = False,
    format: NeighborListFormat = NeighborListFormat.Sparse,
    num_partitions: int = 8,
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
        box: Either a float specifying the size of the box or an array of
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
    assert not fractional_coordinates, "Works only with real coordinates"
    assert format == NeighborListFormat.Sparse, "Works only with sparse neighbor list"
    assert custom_mask_function is None, "Custom masking not implemented"

    is_format_valid(format)
    box = lax.stop_gradient(box)
    r_cutoff = lax.stop_gradient(r_cutoff)
    dr_threshold = lax.stop_gradient(dr_threshold)

    box = jnp.float32(box)

    cutoff = r_cutoff + dr_threshold
    cutoff_sq = cutoff**2
    threshold_sq = (dr_threshold / jnp.float32(2)) ** 2
    metric_sq = _displacement_or_metric_to_metric_sq(displacement_or_metric)

    cell_size = cutoff
    assert jnp.all(cell_size < box / 3.0), "Don't use scan with very few cells"

    def neighbor_list_fn(
        position: jnp.ndarray,
        neighbors: Optional[NeighborList] = None,
        extra_capacity: int = 0,
        **kwargs,
    ) -> NeighborList:
        def neighbor_fn(position_and_error, max_occupancy=None):
            position, err = position_and_error
            N, dim = position.shape
            cl_fn = None
            cl = None
            cell_size = None

            if neighbors is None:  # cl.shape = (nx, ny, nz, cell_capacity, dim)
                cell_size = cutoff
                cl_fn = cell_list(box, cell_size, capacity_multiplier)
                cl = cl_fn.allocate(position, extra_capacity=extra_capacity)
            else:
                cell_size = neighbors.cell_size
                cl_fn = neighbors.cell_list_fn
                if cl_fn is not None:
                    cl = cl_fn.update(position, neighbors.cell_list_capacity)

            err = err.update(PEC.CELL_LIST_OVERFLOW, cl.did_buffer_overflow)
            cl_capacity = cl.cell_capacity

            idx = cl.id_buffer

            cell_idx = [idx]  # shape: (nx, ny, nz, cell_capacity, 1)

            for dindex in _neighboring_cells(dim):
                if onp.all(dindex == 0):
                    continue
                cell_idx += [shift_array(idx, dindex)]

            cell_idx = jnp.concatenate(cell_idx, axis=-2)
            cell_idx = jnp.reshape(cell_idx, (-1, cell_idx.shape[-2]))
            num_cells, considered_neighbors = cell_idx.shape

            particle_cells = get_particle_cells(idx, cl_capacity, N)

            d = partial(metric_sq, **kwargs)
            d = space.map_bond(d)

            # number of particles per partition N_sub
            # np.ceil used to pad last partition with < num_partitions entries
            N_sub = int(np.ceil(N / num_partitions))
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
                # the area of a circle with r=1/3 is 0.34907
                volumetric_factor = 0.34907
            elif dim == 3:
                # the volume of a sphere with r=1/3 is 0.15514
                volumetric_factor = 0.15514

            num_edges_sub = int(
                N_sub * considered_neighbors * volumetric_factor * capacity_multiplier
            )

            def scan_body(carry, input):
                """Compute neighbors over a subset of particles

                The largest object here is of size (N_sub*considered_neighbors), where
                considered_neighbors in 3D is 27 * cell_capacity.
                """

                occupancy = carry
                slice_from = input

                _entries = lax.dynamic_slice(particle_cells, (slice_from,), (N_sub,))
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

                sender_idx = jnp.reshape(sender_idx, (-1,))
                receiver_idx = jnp.reshape(_idx, (-1,))
                dR = d(position[sender_idx], position[receiver_idx])

                mask = (dR < cutoff_sq) & (receiver_idx < N)
                out_idx = N * jnp.ones(receiver_idx.shape, jnp.int32)

                cumsum = jnp.cumsum(mask)
                index = jnp.where(mask, cumsum - 1, considered_neighbors * N - 1)
                receiver_idx = out_idx.at[index].set(receiver_idx)
                sender_idx = out_idx.at[index].set(sender_idx)
                occupancy += cumsum[-1]

                carry = occupancy
                y = jnp.stack(
                    (receiver_idx[:num_edges_sub], sender_idx[:num_edges_sub])
                )
                overflow = cumsum[-1] > num_edges_sub
                return carry, (y, overflow)

            carry = jnp.array(0)
            xs = jnp.array([i * N_sub for i in range(num_partitions)])
            occupancy, (idx, overflows) = lax.scan(
                scan_body, carry, xs, length=num_partitions
            )
            err = err.update(PEC.CELL_LIST_OVERFLOW, overflows.sum())
            idx = idx.transpose(1, 2, 0).reshape(2, -1)

            # sort to enable pruning later
            ordering = jnp.argsort(idx[1])
            idx = idx[:, ordering]

            if max_occupancy is None:
                _extra_capacity = N * extra_capacity
                max_occupancy = int(occupancy * capacity_multiplier + _extra_capacity)
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
            update_fn = neighbor_list_fn if neighbors is None else neighbors.update_fn
            return NeighborList(
                idx,
                position,
                err.update(PEC.NEIGHBOR_LIST_OVERFLOW, occupancy > max_occupancy),
                cl_capacity,
                max_occupancy,
                format,
                cell_size,
                cl_fn,
                update_fn,
            )  # pytype: disable=wrong-arg-count

        nbrs = neighbors
        if nbrs is None:
            return neighbor_fn((position, PartitionError(jnp.zeros((), jnp.uint8))))

        neighbor_fn = partial(neighbor_fn, max_occupancy=nbrs.max_occupancy)

        d = partial(metric_sq, **kwargs)
        d = jax.vmap(d)

        return lax.cond(
            jnp.any(d(position, nbrs.reference_position) > threshold_sq),
            (position, nbrs.error),
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

    dtype_idx = jnp.arange(0).dtype  # just to get the correct dtype

    def matscipy_wrapper(position, idx_shape, num_particles):
        position = position[:num_particles]

        if position.shape[1] == 2:
            position = np.pad(
                position, ((0, 0), (0, 1)), mode="constant", constant_values=0.5
            )

        edge_list = matscipy_nl(
            "ij", cutoff=r_cutoff, positions=position, cell=box_size, pbc=pbc
        )
        edge_list = np.asarray(edge_list, dtype=dtype_idx)
        if not mask_self:
            # add self connection, which matscipy does not do
            self_connect = np.arange(num_particles, dtype=dtype_idx)
            self_connect = np.array([self_connect, self_connect])
            edge_list = np.concatenate((self_connect, edge_list), axis=-1)

        if edge_list.shape[1] > idx_shape[1]:  # overflow true case
            idx_new = np.asarray(edge_list[:, : idx_shape[1]])
            buffer_overflow = np.array(True)
        else:
            idx_new = np.ones(idx_shape, dtype=dtype_idx) * num_particles_max
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
            neighbors.error.update(PEC.NEIGHBOR_LIST_OVERFLOW, buffer_overflow),
            None,
            None,
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
        edge_list = jnp.asarray(edge_list, dtype=dtype_idx)
        if not mask_self:
            # add self connection, which matscipy does not do
            self_connect = jnp.arange(num_particles, dtype=dtype_idx)
            self_connect = jnp.array([self_connect, self_connect])
            edge_list = jnp.concatenate((self_connect, edge_list), axis=-1)

        # in case this is a (2,M) pair list, we pad with N and capacity_multiplier
        factor = capacity_multiplier * num_particles_max / num_particles
        res = num_particles * jnp.ones(
            (2, round(edge_list.shape[1] * factor + extra_capacity)),
            dtype_idx,
        )
        res = res.at[:, : edge_list.shape[1]].set(edge_list)
        return NeighborList(
            res,
            position,
            PartitionError(jnp.zeros((), jnp.uint8)),
            None,
            None,
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
    """Neighbor lists wrapper. Its arguments are mainly based on the jax-md ones.

    Args:
        displacement: A function `d(R_a, R_b)` that computes the displacement
            between pairs of points.
        box_size: Either a float specifying the size of the box or an array of
            shape `[spatial_dim]` specifying the box size in each spatial dimension.
        r_cutoff: A scalar specifying the neighborhood radius.
        dr_threshold: A scalar specifying the maximum distance particles can move
            before rebuilding the neighbor list.
        backend: The backend to use. Can be one of: 1) ``jaxmd_vmap`` - the default
            jax-md neighbor list which vectorizes the computations. 2) ``jaxmd_scan`` -
            a modified jax-md neighbor list which serializes the search into
            ``num_partitions`` chunks to improve the memory efficiency. 3) ``matscipy``
            - a jit-able implementation with the matscipy neighbor list backend, which
            runs on CPU and takes variable number of particles smaller or equal to
            ``num_particles``.
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
        fractional_coordinates: An optional boolean. Specifies whether positions will
            be supplied in fractional coordinates in the unit cube, :math:`[0, 1]^d`.
            If this is set to True then the `box_size` will be set to `1.0` and the
            cell size used in the cell list will be set to `cutoff / box_size`.
        format: The format of the neighbor list; see the :meth:`NeighborListFormat` enum
            for details about the different choices for formats. Defaults to `Dense`.
        num_particles_max: only used with the ``matscipy`` backend. Based
            on the largest particles system in a dataset.
        num_partitions: only used with the ``jaxmd_scan`` backend
        pbc: only used with the ``matscipy`` backend. Defines the boundary conditions
            for each dimension individually. Can have shape (2,) or (3,).
        **static_kwargs: kwargs that get threaded through the calculation of
            example positions.
    Returns:
        A NeighborListFns object that contains a method to allocate a new neighbor
        list and a method to update an existing neighbor list.
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

    # box_size = np.array([1.0, 1.0])
    # r = np.array([[0.1, 0.1], [0.5, 0.1]])
    # displacement_fn, shift_fn = space.periodic(side=box_size)
    # r_cutoff = 0.45

    # neighbor_fn = neighbor_list(
    #     displacement_fn,
    #     box_size,
    #     r_cutoff=r_cutoff,
    #     backend="jaxmd_vmap",
    #     dr_threshold=r_cutoff * 0.25,
    #     capacity_multiplier=1.25,
    #     mask_self=False,
    #     format=Sparse,
    # )
    # neighbors = neighbor_fn.allocate(r)
    # neighbors = neighbors.update(r)
    # neighbors = neighbors.update(r)

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

    pass
