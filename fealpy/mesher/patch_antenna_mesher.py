from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..decorator import variantmethod
from ..mesh import TetrahedronMesh
import math

try:
    import gmsh
except ImportError:
    raise ImportError("The gmsh package is required for EllipsoidMesher. "
                      "Please install it via 'pip install gmsh'.")
    
class PatchAntennaMesher:
    """
    A mesher for generating mesh of a patch antenna consisting of a spherical shell and a thin rectangular plate.
    
    Parameters
        R : float
            The outer radius of the spherical shell.
        r : float
            The inner radius of the spherical shell.
        sphere_center : tuple of float
            The (x, y, z) coordinates of the center of the spherical shell.
        Lx, Ly : float
            The dimensions of the rectangular plate in the x and y directions.
        t : float
            The thickness of the rectangular plate.
        plate_center : tuple of float
            The (x, y, z) coordinates of the center of the rectangular plate.
        inner_size : tuple of float
            The (lx_in, ly_in) dimensions of the inner region of the plate.
        notches : list of tuple
            The (x, y, sx, sy) definitions of the notches in the plate.
        h_sphere_inner, h_sphere_shell : float
            The mesh sizes for the inner and outer regions of the spherical shell.
        h_plate_inner, h_plate_shell : float
            The mesh sizes for the inner and outer regions of the rectangular plate.
        is_optimize : bool
            Whether to optimize the mesh.
        h_min, h_max : float
            The minimum and maximum mesh sizes.
    
    Remarks:
        recommend_mesh_sizes static method provides recommended mesh sizes based on geometry.
        you can use PatchAntennaMesher.recommend_mesh_sizes(...) to get suggested sizes.
    """
    def __init__(self,
                 R=100.0, r=80.0, sphere_center=(0.0, 0.0, 0.0),
                 Lx=100.0, Ly=100.0, t=1.524, plate_center=(0.0, 0.0, 0.0),
                 inner_size=(53.0, 52.0), notches=None,
                 h_sphere_inner=12.0, h_sphere_shell=12.0,
                 h_plate_inner=1.0, h_plate_shell=2.0,
                 is_optimize=False, h_min=None, h_max=None):
        self.R, self.r = R, r
        self.cx, self.cy, self.cz = sphere_center

        self.Lx, self.Ly, self.t = Lx, Ly, t
        self.px, self.py, self.pz = plate_center

        self.lx_in, self.ly_in = inner_size
        self.notches = notches or []

        self.h_sphere_inner = h_sphere_inner
        self.h_sphere_shell = h_sphere_shell
        self.h_plate_inner = h_plate_inner
        self.h_plate_shell = h_plate_shell
        self.h_min = None if h_min is None else h_min
        self.h_max = None if h_max is None else h_max
        self.is_optimize = bool(is_optimize)

        if not (self.R > self.r > 0.0):
            raise ValueError("需满足 R > r > 0")
        if not (0 < self.lx_in < self.Lx and 0 < self.ly_in < self.Ly and self.t > 0):
            raise ValueError("需满足 0<lx_in<Lx, 0<ly_in<Ly, t>0")

        # 预计算板与内层包围盒
        self.x0 = self.px - self.Lx / 2.0
        self.y0 = self.py - self.Ly / 2.0
        self.z0 = self.pz - self.t / 2.0
        self.xi0 = self.px - self.lx_in / 2.0
        self.yi0 = self.py - self.ly_in / 2.0
        self.xi1 = self.xi0 + self.lx_in
        self.yi1 = self.yi0 + self.ly_in

        # 规范化孔（左下+尺寸）
        self.hole_defs = []
        for (hc_x, hc_y, hsx, hsy) in self.notches:
            if hsx <= 0 or hsy <= 0:
                raise ValueError("notches 中孔尺寸 sx, sy 必须为正数")
            hx0 = hc_x - hsx/2.0
            hy0 = hc_y - hsy/2.0
            if hx0 < self.xi0 - 1e-12 or hx0 + hsx > self.xi1 + 1e-12 or hy0 < self.yi0 - 1e-12 or hy0 + hsy > self.yi1 + 1e-12:
                raise ValueError(f"内凹的部分 ({hc_x},{hc_y},{hsx},{hsy}) 超出内层矩形范围")
            self.hole_defs.append((hx0, hy0, hsx, hsy))

        # 保证薄板在球内
        for (qx, qy, qz) in [(self.x0, self.y0, self.pz), (self.x0+self.Lx, self.y0, self.pz),
                             (self.x0, self.y0+self.Ly, self.pz), (self.x0+self.Lx, self.y0+self.Ly, self.pz)]:
            if math.hypot(math.hypot(qx-self.cx, qy-self.cy), qz-self.cz) > self.R + 1e-9:
                raise ValueError("薄板外轮廓超出球外半径 R，请调整位置或尺寸")

    @staticmethod
    def recommend_mesh_sizes(t, r, R, n_thk=2, plate_shell_ratio=4.0, n_radial_inner=2, n_radial_shell=1):
        """
        Recommend mesh sizes for the patch antenna geometry.
        Parameters
            t : float
                Thickness of the rectangular plate.
            r : float
                Inner radius of the spherical shell.
            R : float
                Outer radius of the spherical shell.
            n_thk : int
                Recommended number of elements through the thickness of the plate.
            plate_shell_ratio : float
                Ratio of outer plate mesh size to inner plate mesh size.
            n_radial_inner : int
                Recommended number of elements in the radial direction of the inner sphere.
            n_radial_shell : int
                Recommended number of elements in the radial direction of the outer shell.
        """
        h_plate_inner = t / float(n_thk)
        h_plate_shell = plate_shell_ratio * h_plate_inner
        h_sphere_inner = r / float(n_radial_inner)
        h_sphere_shell = (R - r) / float(n_radial_shell)
        h_min = 0.6 * h_plate_inner
        h_max = 1.8 * max(h_sphere_inner, h_sphere_shell)
        return h_plate_inner, h_plate_shell, h_sphere_inner, h_sphere_shell, h_min, h_max

    @staticmethod
    def _safe_set_option(name, value):
        try:
            gmsh.option.setNumber(name, value)
        except:
            pass
    
    def _tag_plate_bottom(self, plate_inner_in, plate_shell_in):
        tolZ = max(1e-6, 1e-7 * max(self.Lx, self.Ly, self.R, self.t))
        tolPlane = max(1e-6, 1e-7 * max(self.Lx, self.Ly))
        bottom_surfaces = set()

        def collect(vol_list):
            if not vol_list:
                return
            faces = gmsh.model.getBoundary([(3, v) for v in set(vol_list)], oriented=False, recursive=False)
            for dim, s in faces:
                if dim != 2:
                    continue
                try:
                    _, _, cz_s = gmsh.model.occ.getCenterOfMass(2, s)
                except Exception:
                    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(2, s)
                    cz_s = 0.5 * (zmin + zmax)
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(2, s)
                if abs(zmax - zmin) <= tolPlane and abs(cz_s - self.z0) <= tolZ:
                    bottom_surfaces.add(s)

        collect(plate_inner_in)
        collect(plate_shell_in)

        if bottom_surfaces:
            pg = gmsh.model.addPhysicalGroup(2, sorted(bottom_surfaces))
            gmsh.model.setPhysicalName(2, pg, "SURF_PLATE_BOTTOM")
        gmsh.model.occ.synchronize()
        return bottom_surfaces

    def _tag_plate_inner_top(self, plate_inner_in):
        tolZ = max(1e-6, 1e-7 * max(self.Lx, self.Ly, self.R, self.t))
        tolPlane = max(1e-6, 1e-7 * max(self.Lx, self.Ly))
        inner_top_surfaces = set()

        if plate_inner_in:
            faces = gmsh.model.getBoundary([(3, v) for v in set(plate_inner_in)], oriented=False, recursive=False)
            for dim, s in faces:
                if dim != 2:
                    continue
                try:
                    _, _, cz_s = gmsh.model.occ.getCenterOfMass(2, s)
                except Exception:
                    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(2, s)
                    cz_s = 0.5 * (zmin + zmax)
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(2, s)
                if abs(zmax - zmin) <= tolPlane and abs(cz_s - (self.z0 + self.t)) <= tolZ:
                    inner_top_surfaces.add(s)

        if inner_top_surfaces:
            pg = gmsh.model.addPhysicalGroup(2, sorted(inner_top_surfaces))
            gmsh.model.setPhysicalName(2, pg, "SURF_PLATE_INNER_TOP")
        gmsh.model.occ.synchronize()
        return inner_top_surfaces

    def _pick_notch_pair(self, holes):
        best, best_gap = None, -1.0
        for i in range(len(holes)):
            for j in range(i + 1, len(holes)):
                a, b = holes[i], holes[j]
                if a["x0"] > b["x0"]:
                    a, b = b, a
                gap = b["x0"] - a["x1"]
                if gap <= 1e-9:
                    continue
                overlap_y = max(0.0, min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]))
                min_sy = max(1e-12, min(a["sy"], b["sy"]))
                if overlap_y / min_sy < 0.8:
                    continue
                if gap > best_gap:
                    best_gap = gap
                    best = {"xL": a["x1"], "xR": b["x0"], "gap": gap}
        return best

    def _tag_bridge_head(self, plate_inner_in, plate_shell_in):
        bridge_head_surfaces = set()
        if self.hole_defs and len(self.hole_defs) >= 2 and plate_inner_in and plate_shell_in:
            holes = [{"x0": hx0, "x1": hx0 + hsx, "y0": hy0, "y1": hy0 + hsy, "sy": hsy}
                     for (hx0, hy0, hsx, hsy) in self.hole_defs]
            p = self._pick_notch_pair(holes)
            if p:
                xL, xR, gap = p["xL"], p["xR"], p["gap"]
                faces_inner = set(s for dim, s in gmsh.model.getBoundary([(3, v) for v in set(plate_inner_in)], oriented=False, recursive=False) if dim == 2)
                faces_shell = set(s for dim, s in gmsh.model.getBoundary([(3, v) for v in set(plate_shell_in)], oriented=False, recursive=False) if dim == 2)
                shared = faces_inner & faces_shell

                tolPlaneY = max(1e-6, 1e-4 * self.Ly)
                tolZt = max(1e-6, 1e-3 * self.t)
                tolGapRel = 0.25
                tolEnd = max(1e-6, 1e-3 * max(1.0, abs(xL) + abs(xR)))

                for s in shared:
                    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(2, s)
                    wY = ymax - ymin
                    wZ = zmax - zmin
                    wX = xmax - xmin
                    if abs(wY) > tolPlaneY:
                        continue
                    if abs(wZ - self.t) > tolZt:
                        continue
                    if not ((1.0 - tolGapRel) * gap <= wX <= (1.0 + tolGapRel) * gap):
                        continue
                    if not (abs(xmin - xL) <= 5 * tolEnd and abs(xmax - xR) <= 5 * tolEnd):
                        continue
                    bridge_head_surfaces.add(s)

                if bridge_head_surfaces:
                    pg = gmsh.model.addPhysicalGroup(2, sorted(bridge_head_surfaces))
                    gmsh.model.setPhysicalName(2, pg, "SURF_PLATE_INNER_BRIDGE_HEAD")
        gmsh.model.occ.synchronize()
        return bridge_head_surfaces
    
    def _extract_surface_by_name(self, surface_physical_name: str, nodetags_map, cell, strict=True):
        # 找物理面 id 与其面实体 tags
        pg_id = None
        for dim, tag in gmsh.model.getPhysicalGroups():
            if dim == 2 and gmsh.model.getPhysicalName(dim, tag) == surface_physical_name:
                pg_id = tag
                break
        if pg_id is None:
            if not strict:
                return {
                    "node_ids": bm.array([], dtype=int),
                    "tri_nodes": bm.array([], dtype=int).reshape((0, 3)),
                    "tet_faces_on_surface": [],
                    "edges": bm.array([], dtype=int).reshape((0, 2)),
                    "tet_edges_on_surface": [],
                }
            raise ValueError(f"未找到物理面: {surface_physical_name}")
        surface_tags = gmsh.model.getEntitiesForPhysicalGroup(2, pg_id)

        # 收集该物理面的三角形面
        tris = []
        for st in surface_tags:
            etypes, _, enodes_lists = gmsh.model.mesh.getElements(2, st)
            for i, et in enumerate(etypes):
                t = enodes_lists[i].reshape((-1, 3))
                tris.append(t)

        if not tris:
            if not strict:
                return {
                    "node_ids": bm.array([], dtype=int),
                    "tri_nodes": bm.array([], dtype=int).reshape((0, 3)),
                    "tet_faces_on_surface": [],
                    "edges": bm.array([], dtype=int).reshape((0, 2)),
                    "tet_edges_on_surface": [],
                }
            raise RuntimeError("该物理面未发现单元面")
        
        tris = bm.concat(tris, axis=0)
        # 映射为 0 索引
        tri_nodes = bm.array([[nodetags_map[int(t)] for t in tri] for tri in tris])
        node_ids = sorted(set(tri_nodes.reshape(-1).tolist()))

        # 构造查找集合
        tri_set = {frozenset(t.tolist()) for t in tri_nodes}
        surf_edges_set = set()
        for a, b, c in tri_nodes.tolist():
            for e in (tuple(sorted((a, b))), tuple(sorted((a, c))), tuple(sorted((b, c)))):
                surf_edges_set.add(e)

        tet_faces = [(1, 2, 3),  (0, 3, 2), (0, 1, 3), (0, 2, 1)]
        tet_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        tet_face_on_surface = []
        tet_edge_on_surface = []
        cells = cell  # (NC,4)
        for ti in range(cells.shape[0]):
            tet = cells[ti].tolist()
            # faces
            for fid, f in enumerate(tet_faces):
                fn = [tet[i] for i in f]
                if frozenset(fn) in tri_set:
                    tet_face_on_surface.append((ti, fid))
            # edges
            for eid, e in enumerate(tet_edges):
                en = tuple(sorted([tet[i] for i in e]))
                if en in surf_edges_set:
                    tet_edge_on_surface.append((ti, eid))

        edges = bm.array([list(e) for e in sorted(surf_edges_set)]) if surf_edges_set else bm.array([], dtype=int).reshape((0, 2))
        return {
            "node_index": bm.array(node_ids, dtype=int),
            "tri_nodes": tri_nodes,
            "tet_faces_on_surface": tet_face_on_surface,
            "edges": edges,
            "tet_edges_on_surface": tet_edge_on_surface,
        }

    @variantmethod('tet')
    def init_mesh(self, return_bottom=False, 
                        return_inner_top=False, 
                        return_inner_head=False,
                        gmsh_show_fltk=False):
        """
        Generate the tetrahedral mesh for the patch antenna geometry.
        Parameters:
            return_bottom (bool, optional): Defaults to False.
            return_inner_top (bool, optional): Defaults to False.
            return_inner_head (bool, optional): Defaults to False.
            gmsh_show_fltk (bool, optional): Defaults to False.

        Returns:
            TetrahedronMesh: The generated tetrahedral mesh.
            dict: Information about the bottom surface if return_bottom is True.
            dict: Information about the inner top surface if return_inner_top is True.
            dict: Information about the inner bridge head surface if return_inner_head is True.
        """
        gmsh.initialize()
        try:
            gmsh.model.add("sphere_with_two_layer_plate")

            # 1) 几何：球外/内
            v_outer = gmsh.model.occ.addSphere(self.cx, self.cy, self.cz, self.R)
            v_inner = gmsh.model.occ.addSphere(self.cx, self.cy, self.cz, self.r)

            # 2) 薄板与内层盒/孔
            v_plate = gmsh.model.occ.addBox(self.x0, self.y0, self.z0, self.Lx, self.Ly, self.t)
            v_inner_box = gmsh.model.occ.addBox(self.xi0, self.yi0, self.z0, self.lx_in, self.ly_in, self.t)

            inner_box_parts = [v_inner_box]
            if self.hole_defs:
                tools = [(3, gmsh.model.occ.addBox(hx0, hy0, self.z0, hsx, hsy, self.t)) for (hx0, hy0, hsx, hsy) in self.hole_defs]
                ov, _ = gmsh.model.occ.cut(objectDimTags=[(3, v_inner_box)], toolDimTags=tools,
                                           removeObject=True, removeTool=True)
                gmsh.model.occ.synchronize()
                inner_box_parts = [t for _, t in ov]
                if not inner_box_parts:
                    raise RuntimeError("内层盒减去凹陷后为空，请检查 notches 是否过大或越界")

            # 2.2 分域：内层 plate∩inner_box，外层 plate−inner_box
            ov, _ = gmsh.model.occ.intersect([(3, v_plate)], [(3, t) for t in inner_box_parts],
                                             removeObject=False, removeTool=False)
            gmsh.model.occ.synchronize()
            plate_inner_parts = [t for _, t in ov]

            ov, _ = gmsh.model.occ.cut([(3, v_plate)], [(3, t) for t in inner_box_parts],
                                       removeObject=False, removeTool=False)
            gmsh.model.occ.synchronize()
            plate_shell_parts = [t for _, t in ov]

            # 裁剪到球内
            plate_inner_in, plate_shell_in = [], []
            for tag in plate_inner_parts:
                ov, _ = gmsh.model.occ.intersect([(3, tag)], [(3, v_outer)],
                                                 removeObject=True, removeTool=False)
                gmsh.model.occ.synchronize()
                if ov:
                    plate_inner_in.extend([t for _, t in ov])
            for tag in plate_shell_parts:
                ov, _ = gmsh.model.occ.intersect([(3, tag)], [(3, v_outer)],
                                                 removeObject=True, removeTool=False)
                gmsh.model.occ.synchronize()
                if ov:
                    plate_shell_in.extend([t for _, t in ov])

            if not plate_inner_in and not plate_shell_in:
                raise RuntimeError("薄板裁剪到球内为空，请检查板位置/尺寸是否在球内")

            # 3) 球域分层并去掉板
            ov, _ = gmsh.model.occ.cut([(3, v_outer)], [(3, v_inner)], removeObject=True, removeTool=False)
            gmsh.model.occ.synchronize()
            v_shell_parts = [t for _, t in ov]

            plate_in_tools = [(3, tag) for tag in sorted(set(plate_inner_in + plate_shell_in))]
            if plate_in_tools:
                ov, _ = gmsh.model.occ.cut([(3, v_inner)], plate_in_tools, removeObject=True, removeTool=False)
                gmsh.model.occ.synchronize()
                v_core_parts = [t for _, t in ov]

                v_shell_final = []
                for p in v_shell_parts:
                    ov, _ = gmsh.model.occ.cut([(3, p)], plate_in_tools, removeObject=True, removeTool=False)
                    gmsh.model.occ.synchronize()
                    v_shell_final.extend([t for _, t in ov])
            else:
                v_core_parts = [v_inner]
                v_shell_final = v_shell_parts

            # 4) 物理分组
            if plate_inner_in:
                pg_pin = gmsh.model.addPhysicalGroup(3, list(set(plate_inner_in))); gmsh.model.setPhysicalName(3, pg_pin, "VOL_PLATE_INNER")
            if plate_shell_in:
                pg_psh = gmsh.model.addPhysicalGroup(3, list(set(plate_shell_in))); gmsh.model.setPhysicalName(3, pg_psh, "VOL_PLATE_SHELL")
            if v_core_parts:
                pg_core  = gmsh.model.addPhysicalGroup(3, list(set(v_core_parts)));  gmsh.model.setPhysicalName(3, pg_core,  "VOL_SPHERE_INNER")
            if v_shell_final:
                pg_shell = gmsh.model.addPhysicalGroup(3, list(set(v_shell_final))); gmsh.model.setPhysicalName(3, pg_shell, "VOL_SPHERE_SHELL")

            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

            # 5) 面物理组：底面、内层上表面、桥头小面
            self._tag_plate_bottom(plate_inner_in, plate_shell_in)
            self._tag_plate_inner_top(plate_inner_in)
            self._tag_bridge_head(plate_inner_in, plate_shell_in)

            # 6) 尺寸场：体域限制的常量场 + 球体 Ball 场(不用尺寸场太慢)
            self._safe_set_option("Mesh.CharacteristicLengthFromPoints", 0)
            self._safe_set_option("Mesh.CharacteristicLengthFromCurves", 0)
            self._safe_set_option("Mesh.MeshSizeFromCurvature", 0)
            self._safe_set_option("Mesh.MeshSizeExtendFromBoundary", 0)

            gmshfidsetN = gmsh.model.mesh.field.setNumber
            gmshfidsetNs = gmsh.model.mesh.field.setNumbers
            gmshopsetN = gmsh.option.setNumber
            
            fid_inner_const = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(fid_inner_const, "F", f"{float(self.h_plate_inner)}")
            fid_inner_res = gmsh.model.mesh.field.add("Restrict")
            gmshfidsetN(fid_inner_res, "IField", fid_inner_const)
            if plate_inner_in:
                gmshfidsetNs(fid_inner_res, "VolumesList", list(set(plate_inner_in)))

            fid_shell_const = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(fid_shell_const, "F", f"{float(self.h_plate_shell)}")
            fid_shell_res = gmsh.model.mesh.field.add("Restrict")
            gmshfidsetN(fid_shell_res, "IField", fid_shell_const)
            if plate_shell_in:
                gmshfidsetNs(fid_shell_res, "VolumesList", list(set(plate_shell_in)))

            fid_ball = gmsh.model.mesh.field.add("Ball")
            gmshfidsetN(fid_ball, "VIn", float(self.h_sphere_inner))
            gmshfidsetN(fid_ball, "VOut", float(self.h_sphere_shell))
            gmshfidsetN(fid_ball, "XCenter", float(self.cx))
            gmshfidsetN(fid_ball, "YCenter", float(self.cy))
            gmshfidsetN(fid_ball, "ZCenter", float(self.cz))
            gmshfidsetN(fid_ball, "Radius", float(self.r) + 1e-12)

            f_list = [fid_ball]
            if plate_inner_in: f_list.append(fid_inner_res)
            if plate_shell_in: f_list.append(fid_shell_res)
            fid_min_all = gmsh.model.mesh.field.add("Min")
            gmshfidsetNs(fid_min_all, "FieldsList", f_list)
            gmsh.model.mesh.field.setAsBackgroundMesh(fid_min_all)

            _hmin = float(self.h_min) if self.h_min is not None else 0.7 * self.h_plate_inner
            _hmax = float(self.h_max) if self.h_max is not None else 1.6 * max(self.h_sphere_inner, self.h_sphere_shell)
            gmshopsetN("Mesh.CharacteristicLengthMin", _hmin)
            gmshopsetN("Mesh.CharacteristicLengthMax", _hmax)

            gmshopsetN("Mesh.ElementOrder", 1)
            gmshopsetN("Mesh.Algorithm3D", 4)
            gmshopsetN("Mesh.Optimize", 1 if self.is_optimize else 0)
            gmsh.model.occ.synchronize()

            gmshopsetN("General.Verbosity", 5)
            gmsh.logger.start()
            try:
                gmsh.model.mesh.generate(3)
            except Exception:
                log = "\n".join(gmsh.logger.get())
                with open("gmsh_error.log", "w", encoding="utf-8") as f:
                    f.write(log)
                gmsh.write("debug_model.step")
                gmsh.write("debug_model.geo_unrolled")
                raise
            finally:
                gmsh.logger.stop()

            # 7) 提取网格: 点与四面体单元
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            node_tags_bm = bm.from_numpy(node_tags)
            node_coords_bm = bm.from_numpy(node_coords)
            node = node_coords_bm.reshape((-1, 3))
            nodetags_map = {int(j): i for i, j in enumerate(node_tags_bm)}

            # 获取单元信息
            cell_type = 4  # 四面体单元的类型编号为 4
            cell_tags, cell_connectivity = gmsh.model.mesh.getElementsByType(cell_type)

            # 节点编号映射到单元
            evid = bm.array([nodetags_map[int(j)] for j in cell_connectivity])
            cell = evid.reshape((cell_tags.shape[-1], -1))

            mesh = TetrahedronMesh(node, cell)

            # 8) 可选表面数据
            bottom_info = self._extract_surface_by_name("SURF_PLATE_BOTTOM", nodetags_map, cell, strict=True) if return_bottom else None
            inner_top_info = self._extract_surface_by_name("SURF_PLATE_INNER_TOP", nodetags_map, cell, strict=True) if return_inner_top else None
            inner_head_info = self._extract_surface_by_name("SURF_PLATE_INNER_BRIDGE_HEAD", nodetags_map, cell, strict=False) if return_inner_head else None
            if gmsh_show_fltk:
                gmsh.fltk.run()
            return mesh, bottom_info, inner_top_info, inner_head_info

        finally:
            gmsh.finalize()