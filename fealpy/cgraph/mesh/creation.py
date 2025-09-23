from typing import Type
import importlib

from ..nodetype import CNodeType, PortConf, DataType


def get_mesh_class(mesh_type: str) -> Type:
    m = importlib.import_module(f"fealpy.mesh.{mesh_type}_mesh")
    mesh_class_name = mesh_type[0].upper() + mesh_type[1:] + "Mesh"
    return getattr(m, mesh_class_name)


class Box2d(CNodeType):
    r"""Create a mesh in a box-shaped 2D area.

    Inputs:
        mesh_type (str): Type of mesh to granerate.
        domain (tuple[float, float, float, float], optional): Domain.
        nx (int, optional): Segments on x direction.
        ny (int, optional): Segments on y direction.

    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "矩形网格"
    PATH: str = "网格.构造"
    INPUT_SLOTS = [
        PortConf("mesh_type", DataType.MENU, 0, title="网格类型", default="triangle", items=["triangle", "quadrangle"]),
        PortConf("domain", DataType.NONE, title="区域"),
        PortConf("nx", DataType.INT, title="X 分段数", default=10, min_val=1),
        PortConf("ny", DataType.INT, title="Y 分段数", default=10, min_val=1)
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(mesh_type, domain, nx, ny):
        MeshClass = get_mesh_class(mesh_type)
        kwds = {"nx": nx, "ny": ny}
        if domain is not None:
            kwds["box"] = domain
        return MeshClass.from_box(**kwds)


class SphericalShell3d(CNodeType):
    r"""Create a tetrahedral mesh in a spherical shell region.

    Inputs:
        r1 (float, optional): Inner radius of the spherical shell.
        r2 (float, optional): Outer radius of the spherical shell.
        h (float, optional): Mesh size parameter.
    """
    TITLE: str = "带空腔的球体网格"
    PATH: str = "网格.构造"
    INPUT_SLOTS = [
        PortConf("r1", DataType.FLOAT, title="内半径", default=0.05, min_val=0.0),
        PortConf("r2", DataType.FLOAT, title="外半径", default=0.5, min_val=0.0),
        PortConf("h", DataType.FLOAT, title="网格尺度", default=0.04, min_val=1e-6),
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]
    @staticmethod
    def run(r1, r2, h):
        from fealpy.mesh import TetrahedronMesh
        mesh = TetrahedronMesh.from_spherical_shell(r1=r1, r2=r2, h=h)
        return mesh


class DLDMicrofluidicChipMesh2d(CNodeType):
    r"""Create a mesh in a DLD microfluidic chip-shaped 2D area.

    Inputs:
        init_point X (float, optional): Initial point of the chip.
        init_point Y (float, optional): Initial point of the chip.
        chip_height (float, optional): Height of the chip.
        inlet_length (float, optional): Length of the inlet.
        outlet_length (float, optional): Length of the outlet.
        radius (float, optional): Radius of the micropillars.
        n_rows (int, optional): Number of rows of micropillars.
        n_cols (int, optional): Number of columns of micropillars.
        tan_angle (float, optional): Tangent value of the angle of deflection.
        n_stages (int, optional): Number of periods of micropillar arrays.
        stage_length (float, optional): Length of a single period.
        lc (float, optional): Target mesh size.
    Outputs:
        mesh (MeshType): The mesh object created.
        radius (float): Radius of the micropillars.
        centers (ndarray): Coordinates of the centers of the micropillars.
        inlet_boundary (ndarray): Inlet boundary.
        outlet_boundary (ndarray): Outlet boundary.
        wall_boundary (ndarray): Wall boundary of the channel.
    """
    
    TITLE: str = "二维 DLD 微流芯片网格"
    PATH: str = "网格.构造"
    INPUT_SLOTS = [
        PortConf("init_point_x", DataType.FLOAT, 0, default=0.0, title="初始点 X"),
        PortConf("init_point_y", DataType.FLOAT, 0, default=0.0, title="初始点 Y"),
        PortConf("chip_height", DataType.FLOAT, 0, default=1.0, title="芯片高度"),
        PortConf("inlet_length", DataType.FLOAT, 0, default=0.1, title="入口宽度"),
        PortConf("outlet_length", DataType.FLOAT, 0, default=0.1, title="出口宽度"),
        PortConf("radius", DataType.FLOAT, 0, default=1 / (3 * 4 * 3), title="微柱半径"),
        PortConf("n_rows", DataType.INT, 0, default=8, title="行数"),
        PortConf("n_cols", DataType.INT, 0, default=4, title="列数"),
        PortConf("tan_angle", DataType.FLOAT, 0, default=1/7, title="偏转角正切值"),
        PortConf("n_stages", DataType.INT, 0, default=3, title="微柱阵列周期数"),
        PortConf("stage_length", DataType.FLOAT, 0, default=1.4, title="单周期长度"),
        PortConf("lc", DataType.FLOAT, 0, default=0.02, title="网格尺寸")
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格"),
        PortConf("radius", DataType.FLOAT, title="微柱半径"),
        PortConf("centers", DataType.TENSOR, title="微柱圆心坐标"),
        PortConf("inlet_boundary", DataType.TENSOR, title="入口边界"),
        PortConf("outlet_boundary", DataType.TENSOR, title="出口边界"),
        PortConf("wall_boundary", DataType.TENSOR, title="通道壁面边界")
    ]

    @staticmethod
    def run(**options):
        from fealpy.geometry import DLDMicrofluidicChipModeler
        from fealpy.mesher import DLDMicrofluidicChipMesher
        import gmsh

        options = {
            "init_point" : (options.get("init_point_x"), options.get("init_point_y")),
            "chip_height" : options.get("chip_height"),
            "inlet_length" : options.get("inlet_length"),
            "outlet_length" : options.get("outlet_length"),
            "radius" : options.get("radius"),
            "n_rows" : options.get("n_rows"),
            "n_cols" : options.get("n_cols"),
            "tan_angle" : options.get("tan_angle"),
            "n_stages" : options.get("n_stages"),
            "stage_length" : options.get("stage_length"),
            "lc" : options.get("lc")
        }

        gmsh.initialize()
        modeler = DLDMicrofluidicChipModeler(options)
        modeler.build(gmsh)
        mesher = DLDMicrofluidicChipMesher(options)
        mesher.generate(modeler, gmsh)
        gmsh.finalize()

        return (mesher.mesh, mesher.radius, mesher.centers, mesher.inlet_boundary, 
                mesher.outlet_boundary, mesher.wall_boundary)


class CreateMesh(CNodeType):
    r"""Create a mesh object.This node generates a mesh of the specified type 
    using given node and cell data.

    Inputs:
        mesh_type (str): Type of mesh to granerate.
        Supported values: "triangle", "quadrangle", "tetrahedron", "hexahedron".Default is "edgemesh".
        
        domain (tuple[float, float], optional): Domain.
        node(tensor):Coordinates of mesh nodes.
        cell(tensor):Connectivity of mesh cells.

    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "构造网格"
    PATH: str = "网格.构造"
    INPUT_SLOTS = [
        PortConf("mesh_type", DataType.MENU, 0, title="网格类型", default="triangle", 
                 items=["triangle", "quadrangle", "tetrahedron", "hexahedron", "edge"]),
        PortConf("domain", DataType.NONE, title="区域"),
        PortConf("node", DataType.TENSOR, title="节点坐标"),
        PortConf("cell", DataType.TENSOR, title="单元")
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(mesh_type, node, cell):
        MeshClass = get_mesh_class(mesh_type)
        kwds = {"node": node, "cell": cell}
        return MeshClass(**kwds)


class Box3d(CNodeType):
    r"""Create a mesh in a box-shaped 3D area.

    Inputs:
        mesh_type (str): Type of mesh to granerate.
        domain (tuple[float, float, float, float, float, float], optional): Domain.
        nx (int, optional): Segments on x direction.
        ny (int, optional): Segments on y direction.
        nz (int, optional): Segments on z direction.

    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "长方体网格"
    PATH: str = "网格.构造"
    INPUT_SLOTS = [
        PortConf("mesh_type", DataType.MENU, 0, title="网格类型", default="tetrahedron", items=["tetrahedron", "hexahedron"]),
        PortConf("domain", DataType.NONE, title="区域"),
        PortConf("nx", DataType.INT, title="X 分段数", default=10, min_val=1),
        PortConf("ny", DataType.INT, title="Y 分段数", default=10, min_val=1),
        PortConf("nz", DataType.INT, title="Z 分段数", default=10, min_val=1)
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(mesh_type, domain, nx, ny, nz):
        MeshClass = get_mesh_class(mesh_type)
        kwds = {"nx": nx, "ny": ny, "nz": nz}
        if domain is not None:
            kwds["box"] = domain
        return MeshClass.from_box(**kwds)    


class CircleMesh(CNodeType):
    r"""Generate a triangular mesh within a 2D circular domain.

    Inputs:
        X (float): X-coordinate of the circle center.
        Y (float): Y-coordinate of the circle center.
        radius (float): Radius of the circle.
        h (float): Mesh density parameter. Smaller values produce finer meshes.

    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "二维圆形网格" 
    PATH: str = "网格.构造"
    INPUT_SLOTS = [
        PortConf("mesh_type", DataType.MENU, 0, title="网格类型", default="triangle", items=["triangle"]),
        PortConf("domain", DataType.NONE, title="区域"),
        PortConf("X", DataType.FLOAT, title="圆心X坐标"),
        PortConf("Y", DataType.FLOAT, title="圆心Y坐标"),
        PortConf("radius", DataType.FLOAT, title="圆半径"),
        PortConf("h", DataType.FLOAT, title="网格密度")
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(mesh_type, X, Y, radius, h):
        MeshClass = get_mesh_class(mesh_type)
    
        kwds = {"h": h}
        mesh = MeshClass.from_unit_circle_gmsh(**kwds)
        node = mesh.entity('node')

        new_node = radius * node
        new_node[:,0] = new_node[:, 0] + X
        new_node[:,1] = new_node[:, 1] + Y
        mesh.node = new_node

        return mesh


class Cylinder3d(CNodeType):
    r"""Create a mesh in a cylinder-shaped 3D area.

    Inputs:
        radius (float): Radius of the cylinder.
        height (float): Height of the cylinder.
        lc (float, optional): Target mesh size.
    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "圆柱体网格"
    PATH: str = "网格.构造"
    INPUT_SLOTS = [
        PortConf("mesh_type", DataType.MENU, 0, title="网格类型", default="tetrahedron", items=["tetrahedron"]),
        PortConf("radius", DataType.FLOAT, 0, title="圆柱体半径", default=1.0, min_val=1e-6),
        PortConf("height", DataType.FLOAT, 0, title="圆柱体高度", default=2.0, min_val=1e-6),
        PortConf("lc", DataType.FLOAT, 0, title="网格尺寸", default=0.2, min_val=1e-6)
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格"),
    ]

    @staticmethod
    def run(mesh_type, radius, height, lc): 
        import matplotlib.pyplot as plt
        MeshClass = get_mesh_class(mesh_type)
        mesh = MeshClass.from_cylinder_gmsh(radius=radius, height=height, lc=lc)

        return mesh