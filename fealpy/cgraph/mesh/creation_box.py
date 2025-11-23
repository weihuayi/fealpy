
from ..nodetype import CNodeType, PortConf, DataType
from .utils import get_mesh_class

__all__ = [
    "Int1d",
    "Box2d",
    "Box3d",
    "SquareHole",
    "CubeSphericalHole",
    "BoxMinusCylinder"
]


class Int1d(CNodeType):
    r"""Create a mesh in a box-shaped 2D area.

    Inputs:
        mesh_type (str): Type of mesh to granerate.
        domain (tuple[float, float, float, float], optional): Domain.
        n (int, optional): Segments on x direction.

    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "区间网格"
    PATH: str = "网格.构造"
    DESC: str = """在区间区网格内按照水平方向上的单元分段数生成均匀网格。
                该节点通过接受一个区间区域的横坐标，和 x 方向的上的分段整数，构造均匀的三角形或四边形网格。
                使用例子：创建一个二元列表或者数组[x_0 , x_1]节点, 
                其描述了一个区间边界的两个顶点分布，
                将该节点连接到输入上，并输入一个两个整型数 int :  nx 描述了 x 方向上的单元分段数（注意不是节点数），
                将该节点连接到输出即可查看网格构造效果。
                """
    INPUT_SLOTS = [
        PortConf("mesh_type", DataType.MENU, 0, title="网格类型", default="interval", items=["interval"]),
        PortConf("interval", DataType.NONE, title="区域"),
        PortConf("nx", DataType.INT, title="分段数", default=10, min_val=1),
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(mesh_type, interval, nx):
        MeshClass = get_mesh_class(mesh_type)
        kwds = {"nx": nx}
        if interval is not None:
            kwds["interval"] = interval
        return MeshClass.from_interval_domain(**kwds)


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
    DESC: str = """在二维矩形区域内按照两个垂直方向上的单元分段数生成均匀网格。
                该节点通过接受一个矩形区域的横坐标和纵坐标组成的四元数组
                和 x,y 两个方向的上的分段整数，构造均匀的三角形或四边形网格。
                使用例子：创建一个四元列表或者数组[x_0 , x_1, y_0,y_1]节点, 
                其描述了一个矩形边界的四个顶点分布，即 [x_i , y_j] ,i,j = 0,1 这四种组合，
                将该节点连接到输入上，并输入一个两个整型数 int :  nx 和 ny 其描述了 x,y 两个方向上的单元分段数（注意不是节点数），
                将该节点连接到输出即可查看网格构造效果。
                """
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
    DESC: str = """在三维矩体区域内按照三个垂直方向上的单元分段数生成均匀网格。
                该节点通过接受一个矩体区域的三个笛卡尔坐标方向组成的六元数组
                和 x,y,z 三个方向的上的分段整数，构造均匀的四面体或六面体网格。
                使用例子：创建一个六元列表或者数组[x_0 , x_1, y_0,y_1,z_0,z_1]节点,
                其描述了一个矩体边界的八个顶点分布，即 [x_i , y_j,z_k] ,i,j,k = 0,1 这八种组合，
                将该节点连接到输入上，并输入一个两个整型数 int :  nx, ny, nz其描述了 x,y,z 三个方向上的单元分段数（注意不是节点数），
                将该节点连接到输出即可查看网格构造效果。
                """
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
    DESC: str = """在二维矩形区域内挖去一个圆形空腔区域，并按照某个单元尺寸生成三角形网格。
                该节点通过接受一个矩形区域的横坐标和纵坐标组成的四元数组和描述圆形空腔的圆心和半径以及单元尺寸生成非结构三角形网格。
                使用例子：创建一个四元列表或者数组[x_0 , x_1, y_0,y_1]节点, 
                其描述了一个矩形边界的四个顶点分布，即 [x_i , y_j] ,i,j = 0,1 这四种组合，将该节点连接到输入上，
                并分别输入四个浮点型数 X ，Y ，r ，h 表示空腔圆心的两个坐标分量、圆半径和单元尺寸（注意单元尺寸为区域内单元最长边长或者单元外接圆直径），
                将该节点连接到输出即可查看网格构造效果。
                """
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
    DESC: str = """在三维矩体区域内挖去一个球形空腔区域，并按照某个单元尺寸生成四面体网格。
                该节点通过接受一个矩体区域的三个笛卡尔坐标方向组成的六元数组和描述球形空腔的球心和半径以及单元尺寸生成非结构四面体网格。
                使用例子：创建一个六元列表或者数组[x_0 , x_1, y_0,y_1,z_0,z_1]节点, 其描述了一个矩体边界的八个顶点分布， 
                即 [x_i , y_j,z_k] ,i,j,k = 0,1 这八种组合，将该节点连接到输入上，
                并分别输入五个浮点型数 X，Y，Z，r，h 表示空腔球心的三个坐标分量、球半径和单元尺寸（注意单元尺寸为区域内单元最长边长或者单元外接球直径）,
                将该节点连接到输出即可查看网格构造效果。
                """
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
    DESC: str = """  在三维矩体区域内挖去一个圆柱空腔区域，并按照某个单元尺寸生成四面体网格。
                该节点通过接受一个矩体区域的三个笛卡尔坐标方向组成的六元数组
                和描述圆柱空腔的柱心、柱体朝向、半径、高度以及单元尺寸生成非结构四面体网格。
                使用例子：创建一个六元列表或者数组 [x_0 , x_1, y_0,y_1,z_0,z_1]节点, 其描述了一个矩体边界的八个顶点分布，
                即 [x_i , y_j,z_k] ,i,j,k = 0,1 这八种组合，将该节点连接到输入上，并分别输入如下三类浮点数：
                X，Y，Z为柱心坐标分量，ax，ay，az为柱体朝向的三个分量（大小无所谓），cyl_radius，cyl_height，h 为柱体半径、高度和单元尺寸，
                将该节点连接到输出即可查看网格构造效果。
                """
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
