from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh

class StaggeredMeshManager:
    """
    A class to manage staggered mesh configurations for solving PDEs on quadrangle grids.

    This class handles the creation and mapping of three staggered meshes (u, v, p) for 
    velocity and pressure fields in a computational domain. It provides methods to map 
    velocity and pressure fields between cell centers and edges, facilitating numerical 
    computations in finite volume or finite difference methods.

    Attributes:
        nx (int): Number of grid cells in the x-direction.
        ny (int): Number of grid cells in the y-direction.
        hx (float): Grid spacing in the x-direction.
        hy (float): Grid spacing in the y-direction.
        pde (object): PDE model providing boundary conditions and mesh initialization.
        umesh (QuadrangleMesh): Mesh for the u-velocity component.
        vmesh (QuadrangleMesh): Mesh for the v-velocity component.
        pmesh (QuadrangleMesh): Mesh for the pressure field.
        edge_points (ndarray): Barycenters of edges in the pressure mesh.
        cell2edge (ndarray): Mapping from cells to edges in the pressure mesh.
        vertical_edge_indices (ndarray): Indices of vertical edges (for u-velocity).
        horizontal_edge_indices (ndarray): Indices of horizontal edges (for v-velocity).
        u_cell_centers (ndarray): Barycenters of cells in the u-mesh.
        v_cell_centers (ndarray): Barycenters of cells in the v-mesh.
    """

    def __init__(self, pde_model, nx, ny):
        """
        Initialize the StaggeredMeshManager with PDE model and grid parameters.

        Args:
            pde_model (object): PDE model containing mesh initialization and boundary conditions.
            nx (int): Number of grid cells in the x-direction.
            ny (int): Number of grid cells in the y-direction.
        """
        self.nx = nx
        self.ny = ny
        self.hx = 1.0 / nx
        self.hy = 1.0 / ny
        self.pde = pde_model

        # Build three meshes: u, v, p
        self.umesh = QuadrangleMesh.from_box(
            box=[-self.hx / 2, 1 + self.hx / 2, 0, 1],
            nx=nx + 1, ny=ny
        )
        self.vmesh = QuadrangleMesh.from_box(
            box=[0, 1, -self.hy / 2, 1 + self.hy / 2],
            nx=nx, ny=ny + 1
        )
        self.pmesh = pde_model.init_mesh['uniform_qrad'](nx=nx, ny=ny)

        self._init_mesh_mappings()

    def _init_mesh_mappings(self):
        """
        Initialize mappings between meshes, including edge and cell relationships.

        This method computes edge barycenters, cell-to-edge mappings, and classifies edges 
        into vertical (u-velocity) and horizontal (v-velocity) categories. It also stores 
        cell barycenters for the u and v meshes.
        """
        # Edge barycenter
        self.edge_points = self.pmesh.entity_barycenter('edge')
        self.cell2edge = self.pmesh.cell_to_edge()

        # Classify: vertical edges (for u), horizontal edges (for v)
        self.vertical_edge_indices = bm.unique(bm.concatenate([
            self.cell2edge[:, 1], self.cell2edge[:, 3]
        ]))
        self.horizontal_edge_indices = bm.unique(bm.concatenate([
            self.cell2edge[:, 0], self.cell2edge[:, 2]
        ]))

        self.u_cell_centers = self.umesh.entity_barycenter('cell')
        self.v_cell_centers = self.vmesh.entity_barycenter('cell')

    def map_velocity_cell_to_edge(self, u_cell, v_cell, a_p_u, a_p_v):
        """
        Map velocity components from u/v cell centers to pressure mesh edges.

        This method maps u-velocity and v-velocity from their respective cell centers to 
        the corresponding edges (vertical for u, horizontal for v) on the pressure mesh.

        Args:
            u_cell (ndarray): u-velocity values at u-mesh cell centers.
            v_cell (ndarray): v-velocity values at v-mesh cell centers.
            a_p_u (ndarray): Additional parameter for u-velocity (e.g., coefficients).
            a_p_v (ndarray): Additional parameter for v-velocity (e.g., coefficients).

        Returns:
            tuple: (edge_velocity, a_p_edge)
                - edge_velocity (ndarray): Velocity values mapped to pressure mesh edges.
                - a_p_edge (ndarray): Additional parameters mapped to pressure mesh edges.
        """
        VEP = self.edge_points[self.vertical_edge_indices]
        HEP = self.edge_points[self.horizontal_edge_indices]

        # u_cell -> vertical edge
        dist_u = bm.sum((self.u_cell_centers[:, None, :] - VEP[None, :, :])**2, axis=2)
        u_map = bm.argmin(dist_u, axis=1).astype(bm.int32)
        u_edge_idx = self.vertical_edge_indices[u_map]
        # v_cell -> horizontal edge
        dist_v = bm.sum((self.v_cell_centers[:, None, :] - HEP[None, :, :])**2, axis=2)
        v_map = bm.argmin(dist_v, axis=1).astype(bm.int32)
        v_edge_idx = self.horizontal_edge_indices[v_map]

        # Write edge values
        NE = self.pmesh.number_of_edges()
        edge_velocity = bm.zeros(NE)
        a_p_edge = bm.zeros(NE)
        edge_velocity[u_edge_idx] = u_cell
        edge_velocity[v_edge_idx] = v_cell
        a_p_edge[u_edge_idx] = a_p_u
        a_p_edge[v_edge_idx] = a_p_v

        return edge_velocity, a_p_edge

    def map_pressure_edge_to_cell(self, mesh_type, p_edge, p_edge_indices):
        """
        Map pressure values from pressure mesh edges to u or v mesh cell centers.

        Args:
            mesh_type (str): Type of mesh to map to ('u' or 'v').
            p_edge (ndarray): Pressure values on the specified edges.
            p_edge_indices (ndarray): Indices of edges in the pressure mesh.

        Returns:
            tuple: (values_on_cell, edge_to_cell)
                - values_on_cell (ndarray): Pressure values mapped to cell centers.
                - edge_to_cell (ndarray): Mapping of edge indices to cell indices.

        Raises:
            ValueError: If mesh_type is not 'u' or 'v'.
        """
        if mesh_type == 'u':
            cell_centers = self.umesh.entity_barycenter('cell')
        elif mesh_type == 'v':
            cell_centers = self.vmesh.entity_barycenter('cell')
        else:
            raise ValueError("mesh_type must be 'u' or 'v'")

        edge_centers = self.edge_points[p_edge_indices]
        dist = bm.sum((edge_centers[:, None, :] - cell_centers[None, :, :])**2, axis=2)
        edge_to_cell = bm.argmin(dist, axis=1)

        values_on_cell = bm.zeros(cell_centers.shape[0])
        values_on_cell[edge_to_cell] = p_edge

        return values_on_cell, edge_to_cell
    
    def interpolate_pressure_to_edges(self, p):
        """
        Interpolate pressure field from main mesh cells to staggered mesh edges.

        This method interpolates pressure values from the main mesh cell centers to the 
        edges of the pressure mesh, applying boundary conditions where applicable, and 
        then maps these to the u and v mesh cell centers.

        Args:
            p (ndarray): Pressure values at main mesh cell centers.

        Returns:
            tuple: (p_u, p_v)
                - p_u (ndarray): Pressure values mapped to u-mesh cell centers.
                - p_v (ndarray): Pressure values mapped to v-mesh cell centers.
        """
        bd_idx = self.pmesh.boundary_face_index()
        pe2c = self.pmesh.edge_to_cell()[:,:2]
        pepoints = self.pmesh.entity_barycenter('edge')
        # Interpolate pressure to all edge centers
        p_e = (p[pe2c[:, 0]] + p[pe2c[:, 1]]) / 2
        p_e[bd_idx] = self.pde.pressure(pepoints[bd_idx])  # Use analytical boundary values
        # Split into u, v directions
        c2e = self.pmesh.cell_to_edge()
        vertical_edge_indices = bm.unique(bm.concatenate([c2e[:, 1], c2e[:, 3]]))
        horizontal_edge_indices = bm.unique(bm.concatenate([c2e[:, 0], c2e[:, 2]]))
        p_u = p_e[vertical_edge_indices]
        p_v = p_e[horizontal_edge_indices]
        # Map to control volume indices (left and bottom control volumes)
        p_u, _ = self.map_pressure_edge_to_cell('u', p_u, vertical_edge_indices)
        p_v, _ = self.map_pressure_edge_to_cell('v', p_v, horizontal_edge_indices)
        return p_u, p_v