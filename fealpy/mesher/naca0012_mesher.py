from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..decorator import variantmethod
from ..mesh import TriangleMesh

try:
    import gmsh
except ImportError:
    raise ImportError("The gmsh package is required for EllipsoidMesher. "
                      "Please install it via 'pip install gmsh'.")


class  NACA0012Mesher:
    """
    A mesher for generating mesh of a NACA 0012 airfoil within a rectangular box.

    Parameters
    naca_points : array_like
        The coordinates of the NACA 0012 airfoil points,
        the point list must be continued,
        and the last point should be the trailing edge point.
    box : tuple
        The bounding box defined as (x_min, x_max, y_min, y_max).
    singular_points : array_like, optional
        Points where mesh refinement is needed, e.g., leading and trailing edges.
    """
    def __init__(self, naca_points:TensorLike, box=(-0.5, 1.5, -0.3, 0.3), singular_points:TensorLike=None):
        self.box = box
        self.naca_points = bm.array(naca_points, dtype=bm.float64)
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

    @variantmethod('tri')
    def init_mesh(self, h=0.05, singular_h=None, is_quad = 0,
                  thickness=0.005, ratio=2.4, size=0.001) -> TriangleMesh:
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
        f = gmsh.model.mesh.field.add('BoundaryLayer')
        gmsh.model.mesh.field.setNumbers(f, 'CurvesList', self.naca_line_tags)
        gmsh.model.mesh.field.setNumber(f, 'Size', h / 50)
        gmsh.model.mesh.field.setNumber(f, 'Ratio', 2.4)
        gmsh.model.mesh.field.setNumber(f, 'Quads', is_quad)
        gmsh.model.mesh.field.setNumber(f, 'Thickness', h / 10)
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
        node_tags, node, _ = gmsh.model.mesh.getNodes()
        node = bm.array(node, dtype=bm.float64).reshape(-1, 3)[:, :2]
        element_types, element_tags, cell = gmsh.model.mesh.getElements(2)
        cell = bm.array(cell[0], dtype=bm.int64).reshape(-1, 3) - 1

        gmsh.finalize()
        return TriangleMesh(node, cell)

