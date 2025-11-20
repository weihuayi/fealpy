
from ..nodetype import CNodeType, PortConf, DataType
from .utils import get_mesh_class

__all__ = ["CreateMesh", "DLDMicrofluidicChipMesh2d", "DLDMicrofluidicChipMesh3d",
           "TrussTowerMesh"]


class CreateMesh(CNodeType):
    r"""Create a mesh object.This node generates a mesh of the specified type 
    using given node and cell data.

    Inputs:
        mesh_type (str): Type of mesh to granerate.
        Supported values: "triangle", "quadrangle", "tetrahedron", "hexahedron".Default is "edgemesh".
        node(tensor):Coordinates of mesh nodes.
        cell(tensor):Connectivity of mesh cells.

    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "构造网格"
    PATH: str = "网格.构造"
    DESC: str = """从网格点坐标(node)和单元数据(cell)直接生成网格对象。
                该节点直接引用网格点坐标和单元数据张量，并将其解释为网格。
                使用例子：通过两个“数据.张量”节点分别创建网格点坐标张量和单元数据张量，连接到该节点的相应输入上，
                再将该节点连接到输出，即可查看网格构造效果。
                """
    INPUT_SLOTS = [
        PortConf("mesh_type", DataType.MENU, 0, title="网格类型", default="edge", 
                 items=["triangle", "quadrangle", "tetrahedron", "hexahedron", "edge"]),
        PortConf("node", DataType.TENSOR, 1, title="节点坐标"),
        PortConf("cell", DataType.TENSOR, 1, title="单元")
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(mesh_type, node, cell):
        MeshClass = get_mesh_class(mesh_type)
        kwds = {"node": node, "cell": cell}
        return MeshClass(**kwds)


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
        mesh (Mesh): The mesh object created.
        radius (float): Radius of the micropillars.
        centers (tensor): Coordinates of the centers of the micropillars.
        inlet_boundary (tensor): Inlet boundary.
        outlet_boundary (tensor): Outlet boundary.
        wall_boundary (tensor): Wall boundary of the channel.
    """
    TITLE: str = "二维 DLD 微流芯片网格"
    PATH: str = "网格.构造"
    DESC: str = """该节点生成二维DLD微流控芯片的网格剖分, 依据几何与周期参数自动构建微柱
                阵列及流道边界，为微流控芯片数值模拟提供几何与网格基础。"""
    INPUT_SLOTS = [
        PortConf("init_point_x", DataType.FLOAT, 1, default=0.0, title="初始点 X"),
        PortConf("init_point_y", DataType.FLOAT, 1, default=0.0, title="初始点 Y"),
        PortConf("chip_height", DataType.FLOAT, 1, default=1.0, title="芯片长度"),
        PortConf("inlet_length", DataType.FLOAT, 1, default=0.1, title="入口宽度"),
        PortConf("outlet_length", DataType.FLOAT, 1, default=0.1, title="出口宽度"),
        PortConf("radius", DataType.FLOAT, 1, default=1 / (3 * 4 * 3), title="微柱半径"),
        PortConf("n_rows", DataType.INT, 1, default=8, title="行数"),
        PortConf("n_cols", DataType.INT, 1, default=4, title="列数"),
        PortConf("tan_angle", DataType.FLOAT, 1, default=1/7, title="偏转角正切值"),
        PortConf("n_stages", DataType.INT, 1, default=3, title="微柱阵列周期数"),
        PortConf("stage_length", DataType.FLOAT, 1, default=1.4, title="单周期长度"),
        PortConf("lc", DataType.FLOAT, 1, default=0.02, title="网格尺寸")
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


class DLDMicrofluidicChipMesh3d(CNodeType):
    r"""Generate a 3D mesh for a DLD (Deterministic Lateral Displacement) microfluidic chip.

    Inputs:
        init_point_x (float): X-coordinate of the initial reference point.
        init_point_y (float): Y-coordinate of the initial reference point.
        chip_height (float): Total height (length) of the chip domain.
        inlet_length (float): Inlet channel width.
        outlet_length (float): Outlet channel width.
        thickness (float): Chip thickness (z-direction dimension).
        radius (float): Radius of each micropillar.
        n_rows (int): Number of micropillar rows in the array.
        n_cols (int): Number of micropillar columns in the array.
        tan_angle (float): Tangent of the DLD array inclination angle (defines lateral shift).
        n_stages (int): Number of periodic stages (DLD array periods).
        stage_length (float): Length of one periodic stage in the array.
        lc (float): Characteristic mesh size (element size).

    Outputs:
        mesh (Mesh): The generated 3D mesh of the microfluidic chip.
        thickness (float): The effective chip thickness used for meshing.
        radius (float): The micropillar radius used in the geometry.
        centers (Tensor): Coordinates of the micropillar centers.
        inlet_boundary (Tensor): Node or face data defining the inlet boundary.
        outlet_boundary (Tensor): Node or face data defining the outlet boundary.
        wall_boundary (Tensor): Node or face data defining the channel wall boundaries.
    """
    TITLE: str = "三维 DLD 微流芯片网格"
    PATH: str = "网格.构造"
    DESC: str = """该节点生成三维DLD微流控芯片的网格剖分, 依据几何与周期参数自动构建微柱
                阵列及流道边界，为微流控芯片数值模拟提供几何与网格基础。"""
    INPUT_SLOTS = [
        PortConf("init_point_x", DataType.FLOAT, 1, default=0.0, title="初始点 X"),
        PortConf("init_point_y", DataType.FLOAT, 1, default=0.0, title="初始点 Y"),
        PortConf("chip_height", DataType.FLOAT, 1, default=1.0, title="芯片长度"),
        PortConf("inlet_length", DataType.FLOAT, 1, default=0.2, title="入口宽度"),
        PortConf("outlet_length", DataType.FLOAT, 1, default=0.2, title="出口宽度"),
        PortConf("thickness", DataType.FLOAT, 1, default=0.1, title="芯片厚度"),
        PortConf("radius", DataType.FLOAT, 1, default=1 / (3 * 5), title="微柱半径"),
        PortConf("n_rows", DataType.INT, 1, default=3, title="行数"),
        PortConf("n_cols", DataType.INT, 1, default=3, title="列数"),
        PortConf("tan_angle", DataType.FLOAT, 1, default=1/7, title="偏转角正切值"),
        PortConf("n_stages", DataType.INT, 1, default=2, title="微柱阵列周期数"),
        PortConf("stage_length", DataType.FLOAT, 1, default=1.4, title="单周期长度"),
        PortConf("lc", DataType.FLOAT, 1, default=0.02, title="网格尺寸")
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格"),
        PortConf("thickness", DataType.FLOAT, title=""),
        PortConf("radius", DataType.FLOAT, title="微柱半径"),
        PortConf("centers", DataType.TENSOR, title="微柱圆心坐标"),
        PortConf("inlet_boundary", DataType.TENSOR, title="入口边界"),
        PortConf("outlet_boundary", DataType.TENSOR, title="出口边界"),
        PortConf("wall_boundary", DataType.TENSOR, title="通道壁面边界")
    ]

    @staticmethod
    def run(**options):
        from fealpy.geometry import DLDMicrofluidicChipModeler3D
        from fealpy.mesher import DLDMicrofluidicChipMesher3D
        import gmsh

        options = {
            "init_point" : (options.get("init_point_x"), options.get("init_point_y")),
            "chip_height" : options.get("chip_height"),
            "inlet_length" : options.get("inlet_length"),
            "outlet_length" : options.get("outlet_length"),
            "thickness": options.get("thickness"),
            "radius" : options.get("radius"),
            "n_rows" : options.get("n_rows"),
            "n_cols" : options.get("n_cols"),
            "tan_angle" : options.get("tan_angle"),
            "n_stages" : options.get("n_stages"),
            "stage_length" : options.get("stage_length"),
            "lc" : options.get("lc")
        }

        gmsh.initialize()
        modeler = DLDMicrofluidicChipModeler3D(options)
        modeler._apply_auto_config()
        modeler.build(gmsh)
        mesher = DLDMicrofluidicChipMesher3D(options)
        mesher.generate(modeler, gmsh)
        gmsh.finalize()

        return (mesher.mesh, mesher.options.get('thickness'),mesher.radius, mesher.centers, mesher.inlet_boundary, 
                mesher.outlet_boundary, mesher.wall_boundary)


class TrussTowerMesh(CNodeType):
    r"""Generate a 3D mesh for a truss tower structure.

    Inputs:
        n_panel (int): Number of vertical panels.
        Lz (float): Total height in the z-direction.
        Wx (float): Half-length in the x-direction.
        Wy (float): Half-length in the y-direction.
        lc (float): Characteristic length for mesh size control.
        ne_per_bar (int): Number of elements per bar.
        face_diag (bool): Whether to include face diagonal bracing.
        
    Outputs:
        mesh (Mesh): The generated 3D truss tower mesh.
    """
    TITLE: str = "桁架塔网格"
    PATH: str = "preprocess.mesher"
    DESC: str = """该节点生成三维桁架塔结构的网格，沿 z 方向构建长条状桁架结构。
            用户可通过设置面板数量、塔身长度以及截面尺寸来控制塔体的整体几何特征，
            并可指定每根杆件的单元划分密度，从而得到具有精细结构的三维桁架网格。
            节点同时支持面内对角加劲杆的自动生成，用于增强塔体的结构稳定性。"""
                
    INPUT_SLOTS = [
        PortConf("n_panel", DataType.INT, 1, desc="沿 z 方向的面板数量（≥1）", title="面板数量", default=19),
        PortConf("Lz", DataType.FLOAT, 1, desc="桁架塔沿 z 方向的总长度", title="总长度", default=19.0),
        PortConf("Wx", DataType.FLOAT, 1, desc="截面矩形的 x 方向半宽度", title="截面宽度", default=0.45),
        PortConf("Wy", DataType.FLOAT, 1, desc="截面矩形的 y 方向半宽度", title="截面高度", default=0.40),
        PortConf("lc", DataType.FLOAT, 1, desc="用于控制网格尺寸的几何特征长度", title="几何点特征长度", default=0.1),
        PortConf("ne_per_bar", DataType.INT, 1, desc="每根杆件沿长度方向划分的单元数量（≥1）", title="每根杆件单元数", default=1),
        PortConf("face_diag", DataType.BOOL, 0, desc="是否在四个侧面加入面内对角线加劲杆件（默认True）", title="面内对角加劲", default=True)
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, desc="生成桁架塔网格", title="网格")
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.csm.mesh.truss_tower import TrussTower

        node, cell = TrussTower.build_truss_3d_zbar(
            n_panel=options.get("n_panel"),
            Lz=options.get("Lz"),
            Wx=options.get("Wx"),
            Wy=options.get("Wy"),
            lc=options.get("lc"),
            ne_per_bar=options.get("ne_per_bar"),
            face_diag=options.get("face_diag"),
            save_msh=None
        )
        
        from fealpy.mesh import EdgeMesh
        mesh = EdgeMesh(node, cell)
        return mesh

class NACA4Mesh2d(CNodeType):
    TITLE: str = "NACA 四位数翼型二维网格"
    PATH: str = "网格.构造"
    DESC: str = """该节点生成二维 NACA4 系列翼型的网格剖分, 依据翼型参数自动构建翼型几何形状及
                流道边界，为翼型流场数值模拟提供几何与网格基础。"""
    INPUT_SLOTS = [
        PortConf("m", DataType.FLOAT, 0, default=0.0, title="最大弯度"),
        PortConf("p", DataType.FLOAT, 0, default=0.0, title="最大弯度位置"),
        PortConf("t", DataType.FLOAT, 0, default=0.12, title="相对厚度"),
        PortConf("c", DataType.FLOAT, 0, default=1.0, title="弦长"),
        PortConf("alpha", DataType.FLOAT, 0, default=0.0, title="攻角"),
        PortConf("N", DataType.INT, 0, default=200, title="翼型轮廓分段数"),
        PortConf("box", DataType.TENSOR, 1, default=(-5.0, 5.0, -5.0, 5.0), title="求解域"),
        PortConf("h", DataType.FLOAT, 0, default=0.02, title="全局网格尺寸"),
        PortConf("thickness", DataType.FLOAT, 0, default=None, title="边界层厚度"),
        PortConf("ratio", DataType.FLOAT, 0, default=2.4, title="边界层增长率"),
        PortConf("size", DataType.FLOAT, 0, default=None, title="翼型附近网格尺寸")
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]
    def run(**options):
        import gmsh
        from fealpy.backend import backend_manager as bm
        from fealpy.backend import TensorLike
        from fealpy.decorator import variantmethod
        from fealpy.mesh import TriangleMesh
        class  NACA4Mesher:
            """
            A mesher for generating mesh of a NACA 0012 airfoil within a rectangular box.

            Parameters
            m : double
                The parameter for the NACA 4-digit airfoil (maximum camber).
            p : double
                The parameter for the NACA 4-digit airfoil (Position of maximum camber).
            t : double
                The parameter for the NACA 4-digit airfoil (maximum thickness).
            c : double
                The chord length of the airfoil.
            alpha ： double
                The angle of attack of the airfoil in degrees.
            N : int
                The number of points to generate for the airfoil.
            box : tuple
                The bounding box defined as (x_min, x_max, y_min, y_max).
            singular_points : array_like, optional
                Points where mesh refinement is needed, e.g., leading and trailing edges.
            """
            def __init__(self, m=0.02, p=0.4, t=0.12, c = 1.0, alpha=0, N=50,
                        box=(-0.5, 1.5, -0.3, 0.3), singular_points:TensorLike=None):
                self.box = box
                self.naca_points = self.get_naca4_points(m, p, t, c, N)
                if alpha != 0:
                    theta = alpha / 180.0 * bm.pi
                    rotation_matrix = bm.array([[bm.cos(theta), -bm.sin(theta)],
                                                [bm.sin(theta),  bm.cos(theta)]])
                    # self.naca_points = bm.dot(self.naca_points, rotation_matrix.T)
                    # --- Step 1: 计算弦线中点 ---
                    chord_center = bm.array([c / 2, 0.0])

                    # --- Step 2: 平移到原点 ---
                    shifted_points = self.naca_points - chord_center

                    # --- Step 3: 旋转 ---
                    rotated_points = bm.dot(shifted_points, rotation_matrix.T)

                    # --- Step 4: 平移回原位置 ---
                    self.naca_points = rotated_points + chord_center
                if singular_points is not None:
                    self.singular_points = bm.array(singular_points, dtype=bm.float64)
                else:
                    self.singular_points = None
                gmsh.initialize()
                gmsh.option.setNumber("General.Terminal", 0)
                gmsh.model.add("naca0012")

                # 创建大矩形
                box_sphere = gmsh.model.occ.addRectangle(box[0], box[2], 0, box[1]-box[0], box[3]-box[2])
                # 创建 NACA 0012翼型
                point_tags = []
                for p in self.naca_points:
                    point_tags.append(gmsh.model.occ.addPoint(p[0], p[1], 0))
                self.naca_points_tags = point_tags
                line_tags = []
                for i, p in enumerate(point_tags):
                    # 首尾相连：最后一个点连接到第一个点
                    p1 = point_tags[i]
                    p2 = point_tags[(i + 1) % len(point_tags)]  # 模运算确保闭合
                    line = gmsh.model.occ.addLine(p1, p2)
                    line_tags.append(line)
                self.naca_line_tags = line_tags
                halo_curve_loop = gmsh.model.occ.addCurveLoop(line_tags)
                halo_surface = gmsh.model.occ.addPlaneSurface([halo_curve_loop])
                domain_tag, _ = gmsh.model.occ.cut([(2, box_sphere)], [(2, halo_surface)])
                gmsh.model.occ.synchronize()


            def geo_dimension(self) -> int:
                return 2

            def get_naca4_points(self, m, p, t, c=1.0, N=400):
                """
                生成 NACA 4 位数翼型的几何数据
                m: 最大弯度 (max camber)
                p: 最大弯度位置 (location of max camber)
                t: 最大厚度 (max thickness)
                c: 弦长
                N: 采样点数
                """
                # 余弦分布采样（前缘点更密集）
                beta = bm.linspace(0, bm.pi, N)
                x = 0.5 * c * (1 - bm.cos(beta))  # x ∈ [0, c]

                # 厚度分布
                yt = 5 * t * c * (0.2969 * bm.sqrt(x / c) - 0.1260 * (x / c)
                                - 0.3516 * (x / c) ** 2 + 0.2843 * (x / c) ** 3 - 0.1015 * (x / c) ** 4)

                # 弯度线和斜率
                if p != 0 and p != 1:
                    yc = bm.where(x < p * c,
                                m * (x / (p ** 2)) * (2 * p - x / c),
                                m * ((c - x) / ((1 - p) ** 2)) * (1 + x / c - 2 * p))
                    dyc_dx = bm.where(x < p * c,
                                    2 * m / p ** 2 * (p - x / c),
                                    2 * m / (1 - p) ** 2 * (p - x / c))

                if p == 0:
                    yc = m * ((c - x) / ((1 - p) ** 2)) * (1 + x / c - 2 * p)
                    dyc_dx = 2 * m / (1 - p) ** 2 * (p - x / c)

                if p == 1:
                    yc = m * (x / (p ** 2)) * (2 * p - x / c)
                    dyc_dx = 2 * m / p ** 2 * (p - x / c)

                theta = bm.arctan(dyc_dx)

                # 上下表面
                x_u = x - yt * bm.sin(theta)
                y_u = yc + yt * bm.cos(theta)
                x_l = x + yt * bm.sin(theta)
                y_l = yc - yt * bm.cos(theta)

                node_up = bm.flip(bm.stack([x_u, y_u], axis=1), axis=0)
                node_bottom = bm.stack([x_l, y_l], axis=1)[1:]

                # 计算尾部交点
                p0 = node_up[1]
                p1 = node_up[0]
                q0 = node_bottom[-2]
                q1 = node_bottom[-1]
                d0 = p1 - p0
                d1 = q1 - q0
                t_tile = bm.cross((q0 - p0), d1) / bm.cross(d0, d1)
                node_tile = (p0 + t_tile * d0).reshape(1, 2)

                node_airfoil = bm.concat([node_up, node_bottom, node_tile], axis=0)

                return node_airfoil
            
            @variantmethod('tri')
            def init_mesh(self, h=0.05, singular_h=None, is_quad = 0,
                        thickness=None, ratio=None, size=None) -> TriangleMesh:
                """
                Using Gmsh to generate a 2D triangular mesh for a NACA 0012 airfoil within a rectangular box.

                :param h: the global mesh size
                :param singular_h: the local mesh size at singular points
                :param is_quad: is the boundary layer mesh quadrilateral
                :param thickness: the thickness of the boundary layer
                :param ratio: the growth ratio of the boundary layer
                :param size: the initial size of the boundary layer
                :return: the triangle mesh of the NACA 0012 airfoil
                """
                # 设置边界层
                if thickness == None:
                    thickness = h / 10
                if ratio == None:
                    ratio = 2.4
                if size == None:
                    size = h / 50
                f = gmsh.model.mesh.field.add('BoundaryLayer')
                gmsh.model.mesh.field.setNumbers(f, 'CurvesList', self.naca_line_tags)
                gmsh.model.mesh.field.setNumber(f, 'Size', size)
                gmsh.model.mesh.field.setNumber(f, 'Ratio', ratio)
                gmsh.model.mesh.field.setNumber(f, 'Quads', is_quad)
                gmsh.model.mesh.field.setNumber(f, 'Thickness', thickness)
                gmsh.option.setNumber('Mesh.BoundaryLayerFanElements', 7)
                gmsh.model.mesh.field.setNumbers(f, 'FanPointsList', [self.naca_points_tags[-1]])
                gmsh.model.mesh.field.setAsBoundaryLayer(f)
                if self.singular_points is not None:
                    if singular_h is None:
                        singular_h = [h/10]*len(self.singular_points)
                    elif len(singular_h) != len(self.singular_points):
                        raise ValueError("Length of singular_h must match number of singular_points.")
                    # 创建奇异点
                    singular_point_tags = []
                    for i, p in enumerate(self.singular_points):
                        singular_point_tags.append(gmsh.model.occ.addPoint(p[0], p[1], 0, singular_h[i]))
                    gmsh.model.occ.synchronize()
                    # 设置背景网格
                    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
                    # 为奇异点设置局部网格加密
                    singular_point_tags.extend(self.naca_points_tags)
                    singular_h.extend([h/5]*len(self.naca_points_tags))
                    fields = []  # 收集所有 Threshold Field
                    for i, sp in enumerate(singular_point_tags):
                        f_dist = gmsh.model.mesh.field.add("Distance")
                        gmsh.model.mesh.field.setNumbers(f_dist, "PointsList", [sp])

                        f_thresh = gmsh.model.mesh.field.add("Threshold")
                        gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
                        gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", singular_h[i])
                        gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", h)
                        gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", (self.box[3]-self.box[2])/100)
                        gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", (self.box[3]-self.box[2])/2)  # 缩小加密范围

                        fields.append(f_thresh)

                    # 创建 Min Field 合并所有 Threshold Field
                    min_field = gmsh.model.mesh.field.add("Min")
                    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", fields)
                    # 设置背景网格
                    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
                else:
                    # 设置背景网格
                    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

                gmsh.model.mesh.generate(2)
                # gmsh.fltk.run()
                node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
                nodes = bm.array(node_coords, dtype=bm.float64).reshape(-1, 3)[:, :2]
                element_types, element_tags, cell = gmsh.model.mesh.getElements(2)
                cell = bm.array(cell[0], dtype=bm.int64).reshape(-1, 3) - 1
                
                gmsh.finalize()
                return TriangleMesh(nodes, cell)

        
        m = options.get("m", 0.0)
        p = options.get("p", 0.0)
        t = options.get("t", 0.12)
        c = options.get("c", 1.0)
        alpha = options.get("alpha", 0.0)
        N = options.get("N", 200)
        box = options.get("box")
        h = options.get("h", 0.02)
        thickness = options.get("thickness", h/10)
        ratio = options.get("ratio", 2.4)
        size = options.get("size", h/50)
        
        # singular_points = bm.array([[0, 0], [0.97476, 0.260567]], dtype=bm.float64)
        singular_points = bm.array([[0, 0], [1.00, 0.0]], dtype=bm.float64)
        hs = [h/3, h/3] 
        mesher = NACA4Mesher(m , p , t, c, alpha, N, box, singular_points)
        mesh = mesher.init_mesh(h, hs, is_quad=0, thickness = thickness, ratio=ratio, size=size)
        
        return mesh



