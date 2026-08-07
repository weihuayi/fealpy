from typing import Any, Optional, Union, List, Tuple

from fealpy.backend import bm
from fealpy.model import ComputationalModel
from fealpy.material import LinearElasticMaterial

from fealpy.csm.model.linear_elasticity import LinearElasticityPDEDataT
from fealpy.csm.model.model_manager import CSMModelManager

from fealpy.backend import TensorLike
from fealpy.decorator import cartesian
from fealpy.typing import Index, _S


class HydraulicPipeFSIModel(ComputationalModel):
    def __init__(self, options, mesher):
        self.options = options
        self.mesher = mesher
        self.tetra_mesh = mesher.init_mesh()
        self.fluid_mesh = self.extract_fluid_mesh()
        self.interface_mesh = self.extract_interface_mesh()
        self.solid_mesh = self.extract_solid_mesh()
        self._extract_boundary_info()
        self.fluid_rho = options.get('fluid_rho', 1.0)
        self.mu = options.get('mu', 0.003)
        self.solid_rho = options.get('rho', 7800)
        self.E = options.get('E', 2.1e11)
        self.nu = options.get('nu', 0.3)

    def extract_interface_mesh(self):
        from fealpy.mesh import TriangleMesh
        mesh_dict = self.mesher.mesh_data()
        node_id, cell_flat = bm.unique(mesh_dict["interface_tri"], return_inverse=True)
        node = mesh_dict["node"][node_id]
        cell = cell_flat.reshape(-1, 3)
        tri_interface = TriangleMesh(node, cell)
        return tri_interface

    def extract_fluid_mesh(self):
        from fealpy.mesh import TetrahedronMesh
        mesh = self.tetra_mesh
        nodes = mesh.entity('node')
        cells = mesh.entity('cell')
        cell_tags = mesh.celldata['region'] 
        is_fluid_cell = (cell_tags == 1)
        fluid_cells_old_idx = cells[is_fluid_cell]
        unique_nodes, new_cell_nodes = bm.unique(fluid_cells_old_idx, return_inverse=True)
        fluid_nodes = nodes[unique_nodes]
        fluid_cells = new_cell_nodes.reshape(fluid_cells_old_idx.shape)
        fluid_mesh = TetrahedronMesh(fluid_nodes, fluid_cells)
        
        return fluid_mesh

    @cartesian
    def distance_to_wallline(self, p: TensorLike) -> TensorLike:
        R_pipe = 0.5  # 管道半径
        R_bend = 2.8  # 弯管曲率半径

        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]

        # 1. 上游直管 (x <= 0)
        # 轴线在 (y=0, z=0)，点到轴线距离为 sqrt(y^2 + z^2)
        dist_to_axis_up = bm.sqrt(y**2 + z**2)
        d_up = R_pipe - dist_to_axis_up

        # 2. 弯管段 (x > 0 且 y < R_bend)
        # 轴线是以 (0, R_bend) 为圆心，R_bend 为半径的圆弧
        # 在 xy 平面上，点到圆心的距离：
        dist_to_center_xy = bm.sqrt(x**2 + (y - R_bend)**2)
        # 点到圆弧轴线的距离（考虑 z 轴）：
        dist_to_axis_bend = bm.sqrt((dist_to_center_xy - R_bend)**2 + z**2)
        d_bend = R_pipe - dist_to_axis_bend

        # 3. 下游直管 (y >= R_bend)
        # 假设下游沿 y 轴延伸，轴线在 (x=R_bend, z=0)
        # 注意：需根据你 ElbowPipeMesher 的实际生成坐标调整
        dist_to_axis_down = bm.sqrt((x - R_bend)**2 + z**2)
        d_down = R_pipe - dist_to_axis_down

        # 4. 平滑组合 (使用逻辑判断)
        # 修正：d 必须限制最小值为 0，防止数值越界进入壁面内部
        d = bm.where(x <= 0, d_up, 
                        bm.where(y >= R_bend, d_down, d_bend))

        # 限制范围，确保距离在 [0, R_pipe] 之间，防止 SST 模型崩溃
        return bm.maximum(d, 1e-15)
    
    @cartesian
    def is_inlet_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        atol = 1e-12
        on_boundary = (bm.abs(x + 10) < atol)
        return on_boundary
    
    @cartesian
    def is_outlet_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        atol = 1e-12
        on_boundary = (bm.abs(y - 17.8) < atol)
        return on_boundary
    
    @cartesian
    def is_wall_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]

        r = 0.5
        atol = 1e-12
        # d = self.distance_to_wallline(p)
        # on_boundary = (bm.abs(d) < atol)
        node_bc = self.interface_mesh.interpolation_points(p=2)
        d = bm.linalg.norm(p[:, None, :] - node_bc[None, :, :], axis=-1)
        on_boundary = bm.any(d < atol, axis=1)
        
        return on_boundary
    
    # 动量方程
    @cartesian
    def inlet_velocity(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        R = 0.5
        d = self.distance_to_wallline(p)
        u = bm.zeros(p.shape)
        u[..., 0] = 1.224*(1.0 - (0.5 - d)/R)**(1/7)
        u[..., 1] = 0.0
        u[..., 2] = 0.0
        return u
    
    @cartesian
    def outlet_velocity(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        R = 0.5
        d = self.distance_to_wallline(p)
        u = bm.zeros(p.shape)
        u[..., 0] = 0.0
        u[..., 1] = 1.224*(1.0 - (0.5-d)/R)**(1/7)
        u[..., 2] = 0.0
        return u
    
    @cartesian
    def outlet_pressure(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pressure = bm.zeros(x.shape)
        return pressure
    
    @cartesian
    def wall_velocity(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        u = bm.zeros(p.shape)
        return u
    
    @cartesian
    def is_velocity_boundary(self, p: TensorLike) -> TensorLike:
        return self.is_inlet_boundary(p) | self.is_wall_boundary(p)
        # return None
    
    @cartesian
    def is_pressure_boundary(self, p: TensorLike = None) -> TensorLike:
        if p is None:
            return 1
        return self.is_outlet_boundary(p)
        # return 0
    
    @cartesian
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
        result = bm.zeros(p.shape)
        inlet = self.inlet_velocity(p)
        outlet = self.outlet_velocity(p)
        wall = self.wall_velocity(p)
        is_inlet = self.is_inlet_boundary(p)
        is_wall = self.is_wall_boundary(p)
        is_outlet = self.is_outlet_boundary(p)

        result[is_inlet] = inlet[is_inlet]
        result[is_outlet] = outlet[is_outlet]
        result[is_wall] = wall[is_wall]
        return result
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        return self.outlet_pressure(p)
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        result = bm.zeros(p.shape)
        return result
    
    @cartesian
    def pressure_integral_target(self):
        return 0.0
    

    #线弹性方程
    def geo_dimension(self) -> int:
        """Returns the geometric dimension of the domain."""
        return 3
    
    def extract_solid_mesh(self):
        """Initialize and return the tetrahedral mesh for solid domain.
        
        Steps:
        1) Extract solid tetrahedra from the mixed (fluid + solid) mesh.
        2) Keep only nodes referenced by those solid cells.
        3) Remap cell connectivity from global node IDs to compact local IDs (0..N_new-1).
        4) Store global->local node mapping for later boundary-node remapping.
        """
        from fealpy.mesh import TetrahedronMesh
        # Build the full mesh generated by the mesher (contains multiple regions)
        mesh = self.tetra_mesh
        tetra_region = mesh.celldata["region"]
        solid_region = bm.where(tetra_region == 2)[0] # indices of solid tetrahedra
        
        # Extract solid cells and unique nodes
        solid_cell = mesh.cell[solid_region]
        unique_old = bm.unique(solid_cell.reshape(-1))
        n_new = int(unique_old.shape[0])
        n_all = int(mesh.node.shape[0])
        old_to_new = bm.full((n_all,), -1, dtype=bm.int64)
        old_to_new[unique_old] = bm.arange(n_new)
        
        # Save mapping for converting boundary/global node IDs to local mesh IDs later
        self._node_global_to_local = old_to_new
        
        # Create the compact solid tetrahedral mesh
        solid_cell_new = old_to_new[solid_cell]
        solid_node = mesh.node[unique_old]
        solid_mesh = TetrahedronMesh(solid_node, solid_cell_new)

        return solid_mesh
    
    def _extract_boundary_info(self):
        """Extract boundary information from mesh_data for FSI setup."""
        # Get physical name to tag mapping
        mesh_dict = self.mesher.mesh_data()
        self.physical_name_to_dimtag = mesh_dict["physical_name_to_dimtag"]
        
        # Extract physical tags for boundary conditions
        self.solid_inlet_tag = self.physical_name_to_dimtag['solid_inlet_end'][1]
        self.solid_outlet_tag = self.physical_name_to_dimtag['solid_outlet_end'][1]
        self.fsi_interface_tag = self.physical_name_to_dimtag['fsi_interface'][1]
        
        # Extract boundary triangles and markers
        self.boundary_tri = mesh_dict["boundary_tri"]
        self.boundary_tri_marker = mesh_dict["boundary_tri_marker"]
        
        # Extract interface triangles (FSI interface)
        self.interface_tri = mesh_dict["interface_tri"]
        
        # Create mappings for efficient boundary queries
        self._create_boundary_mappings()
        
        # Identify Dirichlet boundary nodes (fixed supports)
        # self.dirichlet_nodes = self._identify_dirichlet_nodes()
        # all_nodes = self.mesh.entity('node')                 # (NN, GD)
        # self.dirichlet_node_coords = all_nodes[self.dirichlet_nodes]  # (Nd, GD)
        dirichlet_global = self._identify_dirichlet_nodes()
        g2l = self._node_global_to_local
        idx = bm.asarray(dirichlet_global, dtype=bm.int64)
        mapped = g2l[idx]
        valid = mapped >= 0
        self.dirichlet_nodes = mapped[valid].tolist()  # 或 bm 转 list，按你类型注解
        all_nodes = self.solid_mesh.entity('node')
        self.dirichlet_node_coords = all_nodes[self.dirichlet_nodes]
        
    def _create_boundary_mappings(self):
        """
        Create mappings between boundary markers and triangle indices.
        """
        self.marker_to_tri_indices = {}
        for i, marker in enumerate(self.boundary_tri_marker):
            if marker not in self.marker_to_tri_indices:
                self.marker_to_tri_indices[marker] = []
            self.marker_to_tri_indices[marker].append(i)
            
    def _identify_dirichlet_nodes(self) -> List[int]:
        """
        Identify nodes on Dirichlet boundaries (fixed supports).
        
        Returns:
            List of node indices on Dirichlet boundaries
        """
        dirichlet_nodes = set()
        
        # Get triangles on solid inlet and outlet ends
        inlet_tri_indices = self.marker_to_tri_indices.get(self.solid_inlet_tag, [])
        outlet_tri_indices = self.marker_to_tri_indices.get(self.solid_outlet_tag, [])
        
        # Collect all nodes on these boundaries
        for idx in inlet_tri_indices:
            dirichlet_nodes.update(self.boundary_tri[idx])
        for idx in outlet_tri_indices:
            dirichlet_nodes.update(self.boundary_tri[idx])
        
        return list(dirichlet_nodes)
    
    def get_dirichlet_nodes(self) -> List[int]:
        """
        Get all nodes on Dirichlet boundaries (fixed supports).
        
        Returns:
            List of node indices that should have fixed displacement
        """
        return self.dirichlet_nodes
    
    def get_fsi_interface(self) -> Tuple[TensorLike, TensorLike]:
        """Get the FSI interface information.
        
        Returns:
            Tuple of (interface triangles, interface node indices)
        """
        # Get all unique nodes on the FSI interface
        if hasattr(self.interface_tri, 'flatten'):
            # For numpy arrays and similar
            fsi_nodes = list(set(self.interface_tri.flatten()))
        else:
            # For lists or other iterables
            fsi_nodes = list(set(node_idx for tri in self.interface_tri for node_idx in tri))
        return self.interface_tri, fsi_nodes
    
    @cartesian
    def body_force(self, p: TensorLike) -> TensorLike:
        """Calculate the body force, such as gravity.
        Parameters:
            p: Coordinates of points inside the element, shape (N, 3), where N is the number of points,
           and 3 represents the x, y, and z coordinates.
    
        Returns:
            The body force density, shape (N, 3), representing the force [Fx, Fy, Fz] at each point.
            For example, under gravity, Fz = -rho * g, 
            where rho is the material density, and g is the gravitational acceleration.
        """
        rho = 7800  
        g = 9.8     
        force = bm.zeros_like(p)
        force[..., 2] = -rho * g  # Fz = -rho * g
        return force 

    @cartesian
    def displacement_bc(self, p: TensorLike) -> TensorLike:
        """Return displacement values on Dirichlet boundaries.
        According to FSI theory, Dirichlet boundaries have fixed displacement (0, 0, 0).

        Parameters:
            p: Points on Dirichlet boundary

        Returns:
            Displacement values (Ux, Uy, Uz) = (0, 0, 0) for fixed boundaries.
        """
        
        return bm.zeros_like(p)
    
    @cartesian
    def is_displacement_boundary(self, p):
        """Determine if the given points lie on a Dirichlet boundary.
        
        Parameters:
            p: Coordinates of points, shape (N, GD), where N is the number of points,
                and GD represents the coordinate dimension (usually 3).
    
        Returns:
            A boolean array indicating whether each point lies on the Dirichlet boundary.
        """
        # p: (N, GD)
        # dirichlet_node_coords: (Nd, GD)
        diff = p[:, None, :] - self.dirichlet_node_coords[None, :, :]
        dist = bm.linalg.norm(diff, axis=-1)         # (N, Nd)
        min_dist = bm.min(dist, axis=1)              # (N,)
        tol = 1e-6  
        return min_dist < tol

    





        