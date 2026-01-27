from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..decorator import variantmethod
from ..mesh import TriangleMesh

try:
    import gmsh
except ImportError:
    raise ImportError("The gmsh package is required for EllipsoidMesher. "
                      "Please install it via 'pip install gmsh'.")


class NACA4Mesher:
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
    def __init__(self, m=0.02, p=0.4, t=0.12, c=1.0, alpha=0, N=50,
                 box=(-0.5, 1.5, -0.3, 0.3), singular_points: TensorLike = None):
        self.box = box
        self.naca_points = self.get_naca4_points(m, p, t, c, N)
        if alpha != 0:
            theta = alpha / 180.0 * bm.pi
            rotation_matrix = bm.array([[bm.cos(theta), -bm.sin(theta)],
                                        [bm.sin(theta), bm.cos(theta)]])
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
        box_sphere = gmsh.model.occ.addRectangle(box[0], box[2], 0, box[1] - box[0], box[3] - box[2])
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
    def init_mesh(self, h=0.05, singular_h=None, is_quad=0,
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
                singular_h = [h / 10] * len(self.singular_points)
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
            singular_h.extend([h / 5] * len(self.naca_points_tags))
            fields = []  # 收集所有 Threshold Field
            for i, sp in enumerate(singular_point_tags):
                f_dist = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(f_dist, "PointsList", [sp])

                f_thresh = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
                gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", singular_h[i])
                gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", h)
                gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", (self.box[3] - self.box[2]) / 100)
                gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", (self.box[3] - self.box[2]) / 2)  # 缩小加密范围

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
