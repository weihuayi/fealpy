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
                  thickness=0.005, ratio=2.4, size=0.001,
              convert_quads_to_tris=True,
              area_tol=1e-8, merge_tol=1e-10) -> TriangleMesh:
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
            singular_point_tags = [
                gmsh.model.occ.addPoint(p[0], p[1], 0, singular_h[i])
                for i, p in enumerate(self.singular_points)
            ]
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
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        nodes = bm.array(node_coords, dtype=bm.float64).reshape(-1, 3)[:, :2]

        element_types, element_tags, element_node_lists = gmsh.model.mesh.getElements(2)

        tri_cells = []
        for etype, nodelist in zip(element_types, element_node_lists):
            nodelist = bm.array(nodelist, dtype=bm.int64)
            if etype == 2:  # Triangle
                tri_cells.append(nodelist.reshape(-1, 3) - 1)
            elif etype == 3 and convert_quads_to_tris:  # Quadrangle -> 2 triangles
                arr = nodelist.reshape(-1, 4) - 1
                t1 = arr[:, [0, 1, 2]]
                t2 = arr[:, [0, 2, 3]]
                tri_cells.extend([t1, t2])
            elif etype == 3 and not convert_quads_to_tris:
                print("[Warning] Quad elements found but not converted!")
            else:
                # Fallback for other element types (only first 3 nodes)
                tri_cells.append(nodelist.reshape(-1, 3) - 1)

        cells = bm.concatenate(tri_cells, axis=0)

        # ---------- Remove zero/small area triangles ----------
        def tri_area(p0, p1, p2):
            return 0.5 * abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]))

        areas = bm.array([tri_area(nodes[i0], nodes[i1], nodes[i2]) for i0, i1, i2 in cells])
        valid_mask = areas > area_tol
        if bm.sum(valid_mask) != cells.shape[0]:
            print(f"[Mesh Cleanup] Removed {cells.shape[0] - bm.sum(valid_mask)} zero/small area triangles.")
            cells = cells[valid_mask]

        # ---------- Merge nearly duplicate nodes ----------
        coords = bm.asarray(nodes)
        scale = 1.0 / max(1e-12, merge_tol)
        keys = bm.round(coords * scale).astype('int64')
        unique_keys, inv = bm.unique(keys, axis=0, return_inverse=True)
        new_coords = unique_keys.astype('float64') / scale
        cells = inv[cells]
        nodes = bm.array(new_coords, dtype=bm.float64)

        gmsh.finalize()
        return TriangleMesh(nodes, cells)

