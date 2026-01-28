from fealpy.backend import TensorLike
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
import gmsh
from fealpy.geometry import DLDMicrofluidicChipModeler3D
from fealpy.mesher import DLDMicrofluidicChipMesher3D


class Cylinder3D():
    def __init__(self, options : dict = None):
        self.eps = 1e-10
        self.mu = 1e-3
        self.rho = 1.0
        self.options = options
        self.mesh = self.init_mesh()
        

    def get_dimension(self) -> int: 
        """Return the geometric dimension of the domain."""
        return 3
    
    # def init_mesh(self):
    #     from fealpy.mesh import TetrahedronMesh
    #     import gmsh
    #     options = {
    #         "box": self.options.get("box", [0, 2.5, 0, 0.41, 0, 0.41]),
    #         "cyl_center": self.options.get("center", [0.5,0.2,0.0]),
    #         "cyl_axis": self.options.get("cyl_axis", [0.0, 0.0, 1.0]),
    #         "cyl_radius": self.options.get("radius", 0.05),
    #         "cyl_height":self.options.get("thickness", 0.10),
    #         "h": self.options.get("lc", 0.1)
    #         }
    #     mesh = TetrahedronMesh.from_box_minus_cylinder(**options)
    #     return mesh

    # def init_mesh(self):
    #     import gmsh
    #     import numpy as np
    #     from fealpy.mesh import TetrahedronMesh

    #     # -----------------------
    #     # Parameters
    #     # -----------------------
    #     L, H, W = 2.2, 0.41, 0.41
    #     xc, yc = 0.2, 0.2
    #     R = 0.05

    #     lc_bulk = 0.08
    #     lc_cyl = 0.03

    #     # -----------------------
    #     # Initialize gmsh
    #     # -----------------------
    #     gmsh.initialize()
    #     gmsh.model.add("cylinder_flow_3d")
    #     occ = gmsh.model.occ

    #     # 强制纯四面体网格（关键）
    #     gmsh.option.setNumber("Mesh.Algorithm3D", 1)   # Delaunay
    #     gmsh.option.setNumber("Mesh.RecombineAll", 0)

    #     # -----------------------
    #     # Geometry
    #     # -----------------------
    #     box = occ.addBox(0, 0, 0, L, H, W)
    #     cyl = occ.addCylinder(xc, yc, 0, 0, 0, W, R)
    #     fluid, _ = occ.cut([(3, box)], [(3, cyl)])

    #     occ.synchronize()

    #     # -----------------------
    #     # Physical surfaces
    #     # -----------------------
    #     surfaces = gmsh.model.getBoundary(fluid, oriented=False)

    #     inlet, outlet, walls, cylinder, front_back = [], [], [], [], []

    #     for dim, tag in surfaces:
    #         x, y, z = gmsh.model.occ.getCenterOfMass(dim, tag)

    #         if abs(x) < 1e-6:
    #             inlet.append(tag)
    #         elif abs(x - L) < 1e-6:
    #             outlet.append(tag)
    #         elif abs(y) < 1e-6 or abs(y - H) < 1e-6:
    #             walls.append(tag)
    #         elif abs(z) < 1e-6 or abs(z - W) < 1e-6:
    #             front_back.append(tag)
    #         else:
    #             cylinder.append(tag)

    #     gmsh.model.addPhysicalGroup(2, inlet, name="inlet")
    #     gmsh.model.addPhysicalGroup(2, outlet, name="outlet")
    #     gmsh.model.addPhysicalGroup(2, walls, name="walls")
    #     gmsh.model.addPhysicalGroup(2, front_back, name="front_back")
    #     gmsh.model.addPhysicalGroup(2, cylinder, name="cylinder")
    #     gmsh.model.addPhysicalGroup(3, [fluid[0][1]], name="fluid")

    #     # -----------------------
    #     # Mesh size field
    #     # -----------------------
    #     gmsh.model.mesh.field.add("Distance", 1)
    #     gmsh.model.mesh.field.setNumbers(1, "FacesList", cylinder)

    #     gmsh.model.mesh.field.add("Threshold", 2)
    #     gmsh.model.mesh.field.setNumber(2, "InField", 1)
    #     gmsh.model.mesh.field.setNumber(2, "SizeMin", lc_cyl)
    #     gmsh.model.mesh.field.setNumber(2, "SizeMax", lc_bulk)
    #     gmsh.model.mesh.field.setNumber(2, "DistMin", 0.05)
    #     gmsh.model.mesh.field.setNumber(2, "DistMax", 0.2)

    #     gmsh.model.mesh.field.setAsBackgroundMesh(2)

    #     # -----------------------
    #     # Generate mesh
    #     # -----------------------
    #     gmsh.model.mesh.generate(3)

    #     # -----------------------
    #     # Extract nodes
    #     # -----------------------
    #     node_tags, node_coords, _ = gmsh.model.mesh.getNodes() 
    #     elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3) 
    #     tri_nodes = elem_node_tags[0].reshape(-1, 3) - 1 # 转为从0开始索引 
    #     node_coords = bm.array(node_coords).reshape(-1, 3)[:, :2] 
    #     tri_nodes = bm.array(tri_nodes, dtype=bm.int32) 

    #     gmsh.fltk.run()

    #     gmsh.finalize()

    #     return TetrahedronMesh(node_tags, node_coords)

    def init_mesh(self):
        options = {
        'init_point': (0.0, 0.0),
        'chip_height': 0.41,
        'inlet_length': 0.2,
        'outlet_length': 0.2,
        'auto_config' : False,
        'thickness': 0.41,
        'radius': 0.05,
        'n_rows': 1,
        'n_cols': 1,
        'tan_angle': 0,
        'n_stages': 1,
        'stage_length': 2.5,
        'lc': 0.02,
        'backend': 'numpy',
        'show_figure': False,
        'space_degree': 3,
        'pbar_log': True,
        'log_level': 'INFO',
        'method': 'Newton',
        'run': 'main',
        'solve': 'direct',
        'apply_bc': 'cylinder',
        'postprocess': 'res',
        'maxit': 1,
        'maxstep': 1000,
        'tol': 1e-10,
        'local_refine': True
        }


        bm.set_backend(options['backend'])

        gmsh.initialize()
        modeler = DLDMicrofluidicChipModeler3D(options)
        modeler._apply_auto_config()
        modeler.build(gmsh)
        mesher = DLDMicrofluidicChipMesher3D(options)
        mesher.generate(modeler, gmsh)
        gmsh.fltk.run()
        gmsh.finalize()

        mesh = mesher.mesh

        return mesh






    @cartesian
    def source(self, p: TensorLike, t) -> TensorLike:
        """Compute exact source """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        result = bm.zeros(p.shape, dtype=bm.float64)
        return result

    @cartesian
    def inlet_velocity(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity."""
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = 80*0.45*y*z*(0.41-y)*(0.41-z)/(0.41**4)
        return result
    
    @cartesian
    def outlet_pressure(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity."""
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        result = bm.zeros(p.shape[0], dtype=bm.float64)
        return result
    
    @cartesian
    def is_inlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        atol = 5e-3
        return bm.abs(p[..., 0]) < atol

    @cartesian
    def is_outlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        x = p[..., 0]
        atol = 0.01
        on_boundary = bm.abs((x - 2.5) <= atol)
        return on_boundary
    @cartesian
    def is_wall_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        y = p[..., 1]
        z = p[..., 2]
        atol = 5e-3
        on_boundary = (
            (bm.abs(y - 0.0) < atol) | (bm.abs(y - 0.41) < atol)|
            (bm.abs(z - 0.0) < atol) | (bm.abs(z - 0.41) < atol))
        return on_boundary
    
    @cartesian
    def is_obstacle_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        radius = 0.05
        atol = 0.01
        # 检查是否接近圆的边界
        on_boundary = (bm.abs((x - 0.5)**2 + (y - 0.2)**2 - radius**2)) < atol
        return on_boundary
    
    @cartesian
    def is_velocity_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        # out = self.is_outlet_boundary(p)
        # return ~out
        return None

    @cartesian
    def is_pressure_boundary(self, p: TensorLike = None) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        # if p is None:
        #     return 1
        # is_out = self.is_outlet_boundary(p)
        # return is_out
        return 0
  
    @cartesian
    def velocity_dirichlet(self, p: TensorLike, t) -> TensorLike:
        """Optional: prescribed velocity on boundary, if needed explicitly."""
        inlet = self.inlet_velocity(p)
        outlet = self.inlet_velocity(p)
        is_inlet = self.is_inlet_boundary(p)
        is_outlet = self.is_outlet_boundary(p)
    
        result = bm.zeros_like(p, dtype=p.dtype)
        result[is_inlet] = inlet[is_inlet]
        result[is_outlet] = outlet[is_outlet]
        
        return result
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike, t) -> TensorLike:
        """Optional: prescribed pressure on boundary (usually for stability)."""
        # outlet = self.outlet_pressure(p)
        # is_outlet = self.is_outlet_boundary(p)

        # result = bm.zeros_like(p[..., 0], dtype=p.dtype)
        # result[is_outlet] = outlet[is_outlet]
        # return result
        return self.outlet_pressure(p)
    
    @cartesian
    def velocity(self, p: TensorLike, t) -> TensorLike:
        """Compute exact solution of velocity."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        return result
    
    @cartesian
    def pressure(self, p: TensorLike, t) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape[0], dtype=p.dtype)
        return result