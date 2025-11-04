
from ..nodetype import CNodeType, PortConf, DataType
from .utils import get_mesh_class

__all__ = [
    "Box2d",
    "Box3d",
    "SquareHole",
    "CubeSphericalHole",
    "BoxMinusCylinder"
]


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
    

class SquareHole(CNodeType):
    r"""Create a mesh in a square with a square hole.

    Inputs:
        mesh_type (str): Type of mesh to granerate.
        domain (tuple[float, float, float, float]): Domain.
        X (float): Center of the square hole in x direction.
        Y (float): Center of the square hole in y direction.
        r (float): Radius of the square hole.
        h (float): The mesh density, the smaller the h, the denser the grid.

    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "带空腔的矩形"
    PATH: str = "网格.构造"
    INPUT_SLOTS = [
        PortConf("mesh_type", DataType.MENU, 0, title="网格类型", default="triangle", items=["triangle"]),
        PortConf("domain", DataType.NONE, title="区域"),
        PortConf("X", DataType.FLOAT, title="空腔X坐标", default=0.5),
        PortConf("Y", DataType.FLOAT, title="空腔Y坐标", default=0.5),
        PortConf("r", DataType.FLOAT, title="空腔半径", default=0.2),
        PortConf("h", DataType.INT, title="网格尺寸", default=0.5)
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(mesh_type, domain, X, Y, r, h):
        MeshClass = get_mesh_class(mesh_type)
        scenter = [X, Y]
        kwds = {"scenter": scenter, "r":r, "h": h}
        if domain is not None:
            kwds["box"] = domain
        
        return MeshClass.from_square_hole(**kwds)
    

class CubeSphericalHole(CNodeType):
    r"""Create a mesh in a cube with a spherical hole.

    Inputs:
        mesh_type (str): Type of mesh to granerate.
        domain (tuple[float, float, float, float, float, float]): Domain.
        X (float): Center of the spherical hole in x direction.
        Y (float): Center of the spherical hole in y direction.
        Z (float): Center of the spherical hole in z direction.
        r (float): Radius of the spherical hole.
        h (float): The mesh density, the smaller the h, the denser the grid

    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "带空腔的长方体"
    PATH: str = "网格.构造"
    INPUT_SLOTS = [
    PortConf("mesh_type", DataType.MENU, 0, title="网格类型", default="tetrahedron", items=["tetrahedron"]),
    PortConf("domain", DataType.NONE, title="区域"),
    PortConf("X", DataType.FLOAT, title="空腔X坐标", default=0.0),
    PortConf("Y", DataType.FLOAT, title="空腔Y坐标", default=0.0),
    PortConf("Z", DataType.FLOAT, title="空腔Z坐标", default=0.0),
    PortConf("r", DataType.FLOAT, title="空腔半径", default=0.5),
    PortConf("h", DataType.INT, title="网格尺寸", default=0.1)
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(mesh_type, domain, X, Y, Z, r, h):
        MeshClass = get_mesh_class(mesh_type)
        scenter = [X, Y, Z]
        kwds = {"scenter": scenter, "r":r, "h": h}
        if domain is not None:
            kwds["box"] = domain
        
        return MeshClass.from_cube_with_spherical_hole(**kwds)


class BoxMinusCylinder(CNodeType):
    r"""Create a mesh in a cube with a spherical hole.

    Inputs:
        mesh_type (str): Type of mesh to granerate.
        domain (tuple[float, float, float, float, float, float]): Domain.
        X (float): Center of the spherical hole in x direction.
        Y (float): Center of the spherical hole in y direction.
        Z (float): Center of the spherical hole in z direction.
        ax (float): Axis direction of the cylinder in x direction.
        ay (float): Axis direction of the cylinder in y direction.
        az (float): Axis direction of the cylinder in z direction.
        cyl_radius (float): Radius of the spherical hole.
        cyl_height (float): Height of the cylinder. If None, the height is set to the maximum length of the box diagonal.
        h (float): The mesh density, the smaller the h, the denser the grid.

    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "带空圆柱的长方体"
    PATH: str = "网格.构造"
    INPUT_SLOTS = [
    PortConf("mesh_type", DataType.MENU, 0, title="网格类型", default="tetrahedron", items=["tetrahedron"]),
    PortConf("domain", DataType.NONE, title="区域"),
    PortConf("X", DataType.FLOAT, title="圆柱X坐标", default=0.5),
    PortConf("Y", DataType.FLOAT, title="圆柱Y坐标", default=0.5),
    PortConf("Z", DataType.FLOAT, title="圆柱Z坐标", default=0.5),
    PortConf("ax", DataType.FLOAT, title="圆柱X轴向", default=1.0),
    PortConf("ay", DataType.FLOAT, title="圆柱Y轴向", default=0.0),
    PortConf("az", DataType.FLOAT, title="圆柱Z轴向", default=0.0),
    PortConf("cyl_radius", DataType.FLOAT, title="圆柱半径", default=0.2),
    PortConf("cyl_height", DataType.FLOAT, title="圆柱高度", default=None),
    PortConf("h", DataType.INT, title="网格尺寸", default=0.1)
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(mesh_type, domain, X, Y, Z, ax, ay, az, cyl_radius, cyl_height, h):
        MeshClass = get_mesh_class(mesh_type)
        cyl_center = [X, Y, Z]
        cyl_axis = [ax, ay, az]
        kwds = {"cyl_center": cyl_center, "cyl_axis":cyl_axis, 
                "cyl_radius":cyl_radius, "cyl_height":cyl_height, "h": h}
        if domain is not None:
            kwds["box"] = domain
        
        return MeshClass.from_box_minus_cylinder(**kwds)
