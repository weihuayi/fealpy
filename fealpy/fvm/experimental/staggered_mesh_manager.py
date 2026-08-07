"""Structured staggered-grid topology helpers for FVM solvers."""

from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh


class StaggeredMeshManager:
    """Build and map the pressure/u/v meshes used by staggered FVM solvers.

    The manager is intentionally limited to structured rectangular
    ``QuadrangleMesh.from_box`` grids.  Its mappings are analytic consequences
    of FEALPy's structured edge numbering, not generic mesh-search operations.
    Values on staggered-grid boundary edges that have no counterpart in the
    other velocity grid must be supplied by Dirichlet boundary data.
    """

    def __init__(self, domain, nx, ny):
        self.nx = nx
        self.ny = ny
        self._mapping_cache = {}
        x_left = domain[0]
        x_right = domain[1]
        y_bottom = domain[2]
        y_top = domain[3]
        self.hx = (x_right-x_left)/ nx
        self.hy = (y_top-y_bottom)/ ny

        self.umesh = QuadrangleMesh.from_box(
            box=[-self.hx / 2 + x_left, self.hx / 2 + x_right, y_bottom, y_top],
            nx=nx + 1, ny=ny
        )
        self.vmesh = QuadrangleMesh.from_box(
            box=[x_left, x_right, -self.hy / 2 + y_bottom, self.hy / 2 + y_top],
            nx=nx, ny=ny + 1
        )
        self.pmesh = QuadrangleMesh.from_box(box=domain, nx=nx, ny=ny)

    def _cached_mapping(self, name, builder):
        if name not in self._mapping_cache:
            self._mapping_cache[name] = builder()
        return self._mapping_cache[name]

    def _index_dtype(self, mesh):
        return mesh.cell_to_edge().dtype

    def _cell_lattice_indices(self, nx, ny, dtype):
        i = bm.repeat(bm.arange(nx, dtype=dtype), ny)
        j = bm.tile(bm.arange(ny, dtype=dtype), (nx,))
        return i, j

    def _edge_lattice_indices(self, nx, ny, dtype):
        i = bm.repeat(bm.arange(nx, dtype=dtype), ny + 1)
        j = bm.tile(bm.arange(ny + 1, dtype=dtype), (nx,))
        return i, j

    def _vertical_edge_id(self, nx, ny, i, j):
        base = 2 * ny + 1
        return bm.where(i < nx, i * base + 2 * j, nx * base + j)

    def _horizontal_edge_id(self, nx, ny, i, j):
        base = 2 * ny + 1
        return bm.where(j < ny, i * base + 2 * j + 1, i * base + 2 * ny)

    def _velocity_component(self, boundary_velocity, points, component, reference):
        if boundary_velocity is None:
            raise ValueError(
                "Boundary velocity is required for staggered edge values "
                "without an interior topological counterpart."
            )

        values = boundary_velocity(points) if callable(boundary_velocity) else boundary_velocity
        if not bm.is_tensor(values):
            values = bm.array(values)
        if len(values.shape) == 0:
            values = bm.full((points.shape[0],), values, dtype=reference.dtype)
        elif len(values.shape) > 1:
            values = values[:, component]
        return bm.array(values, dtype=reference.dtype)

    def get_dof_mapping_ucell2pedge(self):
        def build():
            dtype = self._index_dtype(self.pmesh)
            i, j = self._cell_lattice_indices(self.nx + 1, self.ny, dtype)
            return self._vertical_edge_id(self.nx, self.ny, i, j)

        return self._cached_mapping("ucell2pedge", build)
    
    def get_dof_mapping_vcell2pedge(self):
        def build():
            dtype = self._index_dtype(self.pmesh)
            i, j = self._cell_lattice_indices(self.nx, self.ny + 1, dtype)
            return self._horizontal_edge_id(self.nx, self.ny, i, j)

        return self._cached_mapping("vcell2pedge", build)
    
    def get_dof_mapping_pcell2uedge(self):
        def build():
            dtype = self._index_dtype(self.umesh)
            i, j = self._cell_lattice_indices(self.nx, self.ny, dtype)
            return self._vertical_edge_id(self.nx + 1, self.ny, i + 1, j)

        return self._cached_mapping("pcell2uedge", build)
    
    def get_dof_mapping_pcell2vedge(self):
        def build():
            dtype = self._index_dtype(self.vmesh)
            i, j = self._cell_lattice_indices(self.nx, self.ny, dtype)
            return self._horizontal_edge_id(self.nx, self.ny + 1, i, j + 1)

        return self._cached_mapping("pcell2vedge", build)

    def get_v_edge_for_u_edge_mapping(self):
        """Map v-grid edge values to u-grid edges where topology permits it.

        The returned mask marks the u-grid edges that have a real v-grid edge at
        the same geometric location.  The two outer vertical u-edge columns do
        not have such a counterpart and must be filled from boundary data.
        """
        def build():
            dtype = self._index_dtype(self.vmesh)
            NE = self.umesh.number_of_edges()
            index = bm.zeros(NE, dtype=dtype)
            mask = bm.zeros(NE, dtype=bool)

            iu, ju = self._edge_lattice_indices(self.nx + 1, self.ny, dtype)
            u_horizontal = self._horizontal_edge_id(self.nx + 1, self.ny, iu, ju)
            v_vertical = self._vertical_edge_id(self.nx, self.ny + 1, iu, ju)
            valid_horizontal = bm.full(u_horizontal.shape, True, dtype=bool)

            iv = bm.repeat(bm.arange(1, self.nx + 1, dtype=dtype), self.ny)
            jv = bm.tile(bm.arange(self.ny, dtype=dtype), (self.nx,))
            u_vertical = self._vertical_edge_id(self.nx + 1, self.ny, iv, jv)
            v_horizontal = self._horizontal_edge_id(
                self.nx, self.ny + 1, iv - 1, jv + 1
            )
            valid_vertical = bm.full(u_vertical.shape, True, dtype=bool)

            index = bm.set_at(index, u_horizontal, v_vertical)
            index = bm.set_at(index, u_vertical, v_horizontal)
            mask = bm.set_at(mask, u_horizontal, valid_horizontal)
            mask = bm.set_at(mask, u_vertical, valid_vertical)
            return index, mask

        return self._cached_mapping("v_edge_for_u_edge", build)

    def get_u_edge_for_v_edge_mapping(self):
        """Map u-grid edge values to v-grid edges where topology permits it.

        The returned mask marks the v-grid edges that have a real u-grid edge at
        the same geometric location.  The two outer horizontal v-edge rows do
        not have such a counterpart and must be filled from boundary data.
        """
        def build():
            dtype = self._index_dtype(self.umesh)
            NE = self.vmesh.number_of_edges()
            index = bm.zeros(NE, dtype=dtype)
            mask = bm.zeros(NE, dtype=bool)

            iv, jv = self._cell_lattice_indices(self.nx + 1, self.ny + 1, dtype)
            v_vertical = self._vertical_edge_id(self.nx, self.ny + 1, iv, jv)
            u_horizontal = self._horizontal_edge_id(self.nx + 1, self.ny, iv, jv)
            valid_vertical = bm.full(v_vertical.shape, True, dtype=bool)

            ih = bm.repeat(bm.arange(self.nx, dtype=dtype), self.ny)
            jh = bm.tile(bm.arange(1, self.ny + 1, dtype=dtype), (self.nx,))
            v_horizontal = self._horizontal_edge_id(self.nx, self.ny + 1, ih, jh)
            u_vertical = self._vertical_edge_id(self.nx + 1, self.ny, ih + 1, jh - 1)
            valid_horizontal = bm.full(v_horizontal.shape, True, dtype=bool)

            index = bm.set_at(index, v_vertical, u_horizontal)
            index = bm.set_at(index, v_horizontal, u_vertical)
            mask = bm.set_at(mask, v_vertical, valid_vertical)
            mask = bm.set_at(mask, v_horizontal, valid_horizontal)
            return index, mask

        return self._cached_mapping("u_edge_for_v_edge", build)
    
    def get_dof_mapping_uedge2vedge(self):
        """Compatibility wrapper for u values sampled on v-grid edges.

        Boundary entries without a topological counterpart are placeholder
        indices.  New code should use ``get_u_edge_for_v_edge_mapping`` or
        ``map_u_to_v_edges`` so those entries are filled by boundary data.
        """
        index, _ = self.get_u_edge_for_v_edge_mapping()
        return index
    
    def get_dof_mapping_vedge2uedge(self):
        """Compatibility wrapper for v values sampled on u-grid edges.

        Boundary entries without a topological counterpart are placeholder
        indices.  New code should use ``get_v_edge_for_u_edge_mapping`` or
        ``map_v_to_u_edges`` so those entries are filled by boundary data.
        """
        index, _ = self.get_v_edge_for_u_edge_mapping()
        return index

    def map_v_to_u_edges(self, v_edge, boundary_velocity=None):
        index, mask = self.get_v_edge_for_u_edge_mapping()
        mapped = v_edge[index]
        boundary_values = self._velocity_component(
            boundary_velocity, self.umesh.entity_barycenter("edge"), 1, mapped
        )
        invalid = bm.nonzero(bm.logical_not(mask))[0]
        return bm.set_at(mapped, invalid, boundary_values[invalid])

    def map_u_to_v_edges(self, u_edge, boundary_velocity=None):
        index, mask = self.get_u_edge_for_v_edge_mapping()
        mapped = u_edge[index]
        boundary_values = self._velocity_component(
            boundary_velocity, self.vmesh.entity_barycenter("edge"), 0, mapped
        )
        invalid = bm.nonzero(bm.logical_not(mask))[0]
        return bm.set_at(mapped, invalid, boundary_values[invalid])

    def map_velocity_uvcell_to_pedge(self, u_cell, v_cell, ap_u, ap_v):
        ucell2pedge = self.get_dof_mapping_ucell2pedge()
        vcell2pedge = self.get_dof_mapping_vcell2pedge()

        NE = self.pmesh.number_of_edges()
        p_edge_velocity = bm.zeros(NE, dtype=u_cell.dtype)
        ap_edge = bm.zeros(NE, dtype=ap_u.dtype)
        
        p_edge_velocity = bm.set_at(p_edge_velocity, ucell2pedge, u_cell)
        p_edge_velocity = bm.set_at(p_edge_velocity, vcell2pedge, v_cell)
        ap_edge = bm.set_at(ap_edge, ucell2pedge, ap_u)
        ap_edge = bm.set_at(ap_edge, vcell2pedge, ap_v)

        return p_edge_velocity, ap_edge
    
    def map_pressure_pcell_to_uvedge(self, p):
        ucell2pedge = self.get_dof_mapping_ucell2pedge()
        vcell2pedge = self.get_dof_mapping_vcell2pedge()

        pe2c = self.pmesh.edge_to_cell()[:,:2]
        p_e = (p[pe2c[:, 0]] + p[pe2c[:, 1]]) / 2
        p_u = p_e[ucell2pedge]
        p_v = p_e[vcell2pedge]
        return p_u, p_v
