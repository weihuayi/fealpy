
from ..nodetype import CNodeType, PortConf, DataType
from .utils import get_mesh_class

__all__ = [
    "CircleMesh",
    "Ellipse2d",
    "SphereSurface",
    "Sphere",
    "SphericalShell3d"
]


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
        PortConf("X", DataType.FLOAT, title="圆心X坐标", default=0.0),
        PortConf("Y", DataType.FLOAT, title="圆心Y坐标", default=0.0),
        PortConf("radius", DataType.FLOAT, title="半径", default=1.0),
        PortConf("h", DataType.FLOAT, title="网格尺寸", default=0.5)
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


class Ellipse2d(CNodeType):
    r"""Generate a mesh for an ellipse.

    Inputs:
        a (float): semi-major axis.
        b (float): semi-minor axis.
        center (list): center of the ellipse [cx, cy].
        h (float): mesh size.
        theta (float): rotation angle in radians.

    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "椭圆网格"
    PATH: str = "网格.构造"
    INPUT_SLOTS = [
        PortConf("mesh_type", DataType.MENU, 0, title="网格类型", default="tri", items=["tri", "quad"]),
        PortConf("a", DataType.FLOAT, 1, default=1.0, title="长轴"),
        PortConf("b", DataType.FLOAT, 1, default=1.0, title="短轴"),
        PortConf("x", DataType.FLOAT, 1, default=0.0, title="中心X坐标"),
        PortConf("y", DataType.FLOAT, 1, default=0.0, title="中心Y坐标"),
        PortConf("h", DataType.FLOAT, 1, default=0.1, title="网格尺寸"),
        PortConf("theta", DataType.FLOAT, 1, default=0.0, title="旋转角度(弧度)"),
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格"),
    ]

    @staticmethod
    def run(mesh_type, a, b, x, y, h, theta):
        from fealpy.mesher import EllipseMesher
        MeshClass = EllipseMesher()
        MeshClass.init_mesh.set(mesh_type)
        kwds = {"a": a, "b": b, "x": x, "y": y, "h": h, "theta": theta}
        return MeshClass.init_mesh(**kwds)

    class EllipsoidSurface(CNodeType):
        r"""Generate a mesh for an ellipsoid surface.

        Inputs:
            mesh_type (str): Type of mesh to granerate.
            x (float): the coordinate of the center in x direction.
            y (float): the coordinate of the center in y direction.
            z (float): the coordinate of the center in z direction.
            rx (float): the radii of the ellipsoid along the x axes.
            ry (float): the radii of the ellipsoid along the y axes.
            rz (float): the radii of the ellipsoid along the z axes.
            h (float): mesh size.

        Outputs:
            mesh (MeshType): The mesh object created.
        """
        TITLE: str = "椭球面网格"
        PATH: str = "网格.构造"
        INPUT_SLOTS = [
            PortConf("mesh_type", DataType.MENU, 0, title="网格类型", default="tri", items=["tri", "quad"]),
            PortConf("x", DataType.FLOAT, 1, default=0.0, title="椭球中心的 x 坐标"),
            PortConf("y", DataType.FLOAT, 1, default=0.0, title="椭球中心的 y 坐标"),
            PortConf("z", DataType.FLOAT, 1, default=0.0, title="椭球中心的 z 坐标"),
            PortConf("rx", DataType.FLOAT, 1, default=2.0, title="椭球 x 方向主轴长度"),
            PortConf("ry", DataType.FLOAT, 1, default=1.0, title="椭球 y 方向主轴长度"),
            PortConf("rz", DataType.FLOAT, 1, default=0.5, title="椭球 z 方向主轴长度"),
            PortConf("h", DataType.FLOAT, 1, default=0.1, title="网格尺寸"),
        ]
        OUTPUT_SLOTS = [
            PortConf("mesh", DataType.MESH, title="网格"),
        ]

        @staticmethod
        def run(mesh_type, x, y, z, rx, ry, rz, h):
            from fealpy.mesher import EllipsoidMesher
            ellipsoid_mesher = EllipsoidMesher((x, y, z), (rx, ry, rz))
            if mesh_type == "tri":
                ellipsoid_mesher.init_mesh.set('surface_tri')
            elif mesh_type == "quad":
                ellipsoid_mesher.init_mesh.set('surface_quad')
            else:
                raise ValueError(f"Unsupported mesh type: {mesh_type}")
            mesh = ellipsoid_mesher.init_mesh(h)
            return mesh

    class EllipsoidVolume(CNodeType):
        r"""Generate a mesh for an ellipsoid volume.

        Inputs:
            x (float): the coordinate of the center in x direction.
            y (float): the coordinate of the center in y direction.
            z (float): the coordinate of the center in z direction.
            rx (float): the radii of the ellipsoid along the x axes.
            ry (float): the radii of the ellipsoid along the y axes.
            rz (float): the radii of the ellipsoid along the z axes.
            h (float): mesh size.

        Outputs:
            mesh (MeshType): The mesh object created.
        """
        TITLE: str = "椭球体网格"
        PATH: str = "网格.构造"
        INPUT_SLOTS = [
            PortConf("x", DataType.FLOAT, 1, default=0.0, title="椭球中心的 x 坐标"),
            PortConf("y", DataType.FLOAT, 1, default=0.0, title="椭球中心的 y 坐标"),
            PortConf("z", DataType.FLOAT, 1, default=0.0, title="椭球中心的 z 坐标"),
            PortConf("rx", DataType.FLOAT, 1, default=2.0, title="椭球 x 方向主轴长度"),
            PortConf("ry", DataType.FLOAT, 1, default=1.0, title="椭球 y 方向主轴长度"),
            PortConf("rz", DataType.FLOAT, 1, default=0.5, title="椭球 z 方向主轴长度"),
            PortConf("h", DataType.FLOAT, 1, default=0.1, title="网格尺寸"),
        ]
        OUTPUT_SLOTS = [
            PortConf("mesh", DataType.MESH, title="网格"),
        ]

        @staticmethod
        def run(x, y, z, rx, ry, rz, h):
            from fealpy.mesher import EllipsoidMesher
            ellipsoid_mesher = EllipsoidMesher((x, y, z), (rx, ry, rz))
            ellipsoid_mesher.init_mesh.set('volume')
            mesh = ellipsoid_mesher.init_mesh(h)
            return mesh


class SphereSurface(CNodeType):
    r"""Create a mesh on the surface of a unit sphere.

    Inputs:
        mesh_type (str): Type of mesh to granerate.
        refine (int): Number of mesh refine times.

    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "球面网格"
    PATH: str = "网格.构造"
    DESC: str = "生成单位球面上的网格，输出网格类型与网格加密次数，加密次数越大，网格越密。"
    INPUT_SLOTS = [
        PortConf("mesh_type", DataType.MENU, 0, title="网格类型", default="triangle", items=["triangle", "quadrangle"]),
        PortConf("refine", DataType.INT, 1, title="加密", default=2, min_val=1),
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(mesh_type, refine):
        MeshClass = get_mesh_class(mesh_type)
        kwds = {"refine": refine}
        return MeshClass.from_unit_sphere_surface(**kwds)


class Sphere(CNodeType):
    r"""Create a mesh of a unit sphere.

    Inputs:
        mesh_type (str): Type of mesh to granerate.
        h (float): The mesh density, the smaller the h, the denser the grid.

    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "球体网格"
    PATH: str = "网格.构造"
    DESC: str = "生成单位球体的网格，输入网格类型和网格密度h。h一般为大于0小于1的浮点数，h越小，网格越密。"
    INPUT_SLOTS = [
        PortConf("mesh_type", DataType.MENU, 0, title="网格类型", default="tetrahedron", items=["tetrahedron"]),
        PortConf("h", DataType.FLOAT, 1, title="网格尺寸", default=0.5, min_val=0.1),
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(mesh_type, h):
        MeshClass = get_mesh_class(mesh_type)
        kwds = {"h": h}
        return MeshClass.from_unit_sphere_gmsh(**kwds)


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
        PortConf("h", DataType.FLOAT, title="网格尺寸", default=0.04, min_val=1e-6),
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]
    @staticmethod
    def run(r1, r2, h):
        from fealpy.mesh import TetrahedronMesh
        mesh = TetrahedronMesh.from_spherical_shell(r1=r1, r2=r2, h=h)
        return mesh
