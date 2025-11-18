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
        self.h_bridge_head = self.h_plate_inner / 6.0
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

    def _find_bridge_head_faces_on_mesh(self,mesh):
        """
        在网格上定位“桥头小面”的三角面元索引（mesh.face 的全局下标）。
        规则：
        - 全部顶点 y 在 y = yi0 的容差内
        - 全部顶点 x 落在 [xL, xR]（缺口中心 ± gap/2）的容差内
        """
        if not (self.hole_defs and len(self.hole_defs) >= 2):
            return bm.array([], dtype=bm.int32)

        holes = [{"x0": hx0, "x1": hx0 + hsx, "y0": hy0, "y1": hy0 + hsy, "sy": hsy}
                 for (hx0, hy0, hsx, hsy) in self.hole_defs]
        p = self._pick_notch_pair(holes)
        if not p:
            return bm.array([], dtype=bm.int32)

        xL, xR, gap = p["xL"], p["xR"], p["gap"]
        yi0 = self.yi0

        node = mesh.node
        face = mesh.face

        tolY = max(1e-6, 1e-4 * self.ly_in)
        tolX = max(1e-6, 1e-4 * gap)

        ys = node[face, 1]
        xs = node[face, 0]

        cond_y = (bm.abs(bm.min(ys, axis=1) - yi0) <= tolY) & (bm.abs(bm.max(ys, axis=1) - yi0) <= tolY)
        xmin = bm.min(xs, axis=1); xmax = bm.max(xs, axis=1)
        cond_x = (xmin >= (xL - tolX)) & (xmax <= (xR + tolX))

        idx = bm.nonzero(cond_y & cond_x)[0].astype(bm.int32)
        if idx.size == 0:
            return bm.array([], dtype=bm.int32)
        return bm.unique(idx)
    
    def _find_bridge_head_center_edges_on_mesh(self, mesh, face_idx=None):
        """
        """
        if not (self.hole_defs and len(self.hole_defs) >= 2):
            return bm.array([], dtype=bm.int32)

        holes = [{"x0": hx0, "x1": hx0 + hsx, "y0": hy0, "y1": hy0 + hsy, "sy": hsy}
                 for (hx0, hy0, hsx, hsy) in self.hole_defs]
        p = self._pick_notch_pair(holes)
        if not p:
            return bm.array([], dtype=bm.int32)
        xL, xR, gap = p["xL"], p["xR"], p["gap"]
        xC = 0.5 * (xL + xR)
        yi0 = self.yi0
        # 容差（可按需要收紧/放宽）
        tolY  = max(1e-6, 1e-4 * self.ly_in)
        tolXc = max(1e-6, 1e-4 * gap)

        node = mesh.node   # (N,3)
        face = mesh.face   # (NF,3)
        edge = mesh.edge   # (NE,2)

        # 若未提供小面三角集合，则先用已有函数得到
        if face_idx is None:
            face_idx = self._find_bridge_head_faces_on_mesh(mesh)
        face_idx = bm.asarray(face_idx, dtype=bm.int32)
        N = node.shape[0]
        F = face[face_idx]
        def pair_key(a, b, N_):
            u = bm.minimum(a, b)
            v = bm.maximum(a, b)
            return u * N_ + v

        k1 = pair_key(F[:, 0], F[:, 1], N)
        k2 = pair_key(F[:, 1], F[:, 2], N)
        k3 = pair_key(F[:, 2], F[:, 0], N)
        cand_keys = bm.concat([k1, k2, k3], axis=0)             # (3M,)
        cand_keys = bm.unique(cand_keys)                              # 去重后的 (K,)

        # 4) 全局边集合编码并匹配 (候选键 -> 边全局下标)
        ek_u, ek_v = edge[:, 0], edge[:, 1]
        ekeys = pair_key(ek_u, ek_v, N)                         # (NE,)
        order = bm.argsort(ekeys)
        ekeys_sorted = ekeys[order]
        
        pos = bm.searchsorted(ekeys_sorted, cand_keys)
        mask = (pos < ekeys_sorted.size) & (ekeys_sorted[pos] == cand_keys)
        # 候选边在 edge 中的全局下标
        cand_edge_idx = order[pos[mask]].astype(bm.int32)             # (K',)
        if cand_edge_idx.size == 0:
            return bm.array([], dtype=bm.int32)

        # 5) 在候选边中按中线条件过滤（全部向量化）
        e_ij = edge[cand_edge_idx]                                    # (K',2)
        pi = node[e_ij[:, 0]]                                         # (K',3)
        pj = node[e_ij[:, 1]]                                         # (K',3)

        cond_y = (bm.abs(pi[:, 1] - yi0) <= tolY) & (bm.abs(pj[:, 1] - yi0) <= tolY)
        cond_x = (bm.abs(pi[:, 0] - xC) <= tolXc) & (bm.abs(pj[:, 0] - xC) <= tolXc)
        keep = bm.nonzero(cond_y & cond_x)[0]

        if keep.size == 0:
            return bm.array([], dtype=bm.int32)

        picked = cand_edge_idx[keep]
        return bm.unique(picked.astype(bm.int32))
    
    @staticmethod
    def _mapping_entity_to_ids(entity , nodetags_map):
        mapping = {}
        entity_map = {frozenset(e.tolist()): i for i, e in enumerate(entity)}
        col = entity.shape[1]
        if col == 4:
            etype = 4  # 四面体
        elif col == 3:
            etype = 2  # 三角形
        elif col == 2:
            etype = 1  # 线段
            
        tags, conn = gmsh.model.mesh.getElementsByType(etype)
        if tags.size > 0:
            conn_nodes = [nodetags_map[t] for t in conn]
            conn_nodes = bm.array(conn_nodes).reshape((tags.shape[-1], col))
            for k, etag in enumerate(tags):
                mi = entity_map.get(frozenset(conn_nodes[k].tolist()))
                if mi is not None:
                    mapping[int(etag)] = mi
        return mapping
    
    def _extract_volume_by_name(self, volume_physical_name: str, cell_mapping, strict=True):
        """
        返回该体物理组内所有四面体在 mesh.cell 中的全局编号
        兼容一阶/二阶四面体 (etype 4/11)，逻辑与面提取统一。
        """
        return self._extract_indices_by_name(
            dim=3,
            physical_name=volume_physical_name,
            mapping=cell_mapping,
            allowed_types=(4, 11),
            strict=strict,
        )

    def _extract_surface_by_name(self, surface_physical_name: str, face_mapping, strict=True):
        """
        返回该面物理组内所有三角形在 mesh.face 中的全局编号
        兼容一阶/二阶三角形 (etype 2/9)，逻辑与体提取统一。
        """
        return self._extract_indices_by_name(
            dim=2,
            physical_name=surface_physical_name,
            mapping=face_mapping,
            allowed_types=(2, 9),
            strict=strict,
        )
    def _extract_line_by_name(self, line_physical_name: str, edge_mapping, strict=True):
        """
        返回物理线(如桥头中心竖线)上的所有边在 mesh.edge 中的全局编号。
        兼容一阶/二阶线单元 (etype 1/8)。
        """
        return self._extract_indices_by_name(
            dim=1,
            physical_name=line_physical_name,
            mapping=edge_mapping,
            allowed_types=(1, 8),
            strict=strict,
        )

    def _extract_indices_by_name(self, dim: int, physical_name: str, mapping: dict, allowed_types=(), strict=True):
        """
        从物理组提取元素 tags,并通过 tag->全局索引映射得到全局索引。
        - dim=3, allowed_types=(4,11): 体（四面体）
        - dim=2, allowed_types=(2, 9): 面（三角形）
        - dim=1: allowed_types=(1,8)  线单元 (边)
        返回 bm.array(shape=(k,), dtype=int32)
        """
        if dim not in (1,2, 3):
            raise ValueError("dim 仅支持1, 2 或 3")

        # 1) 找物理组 id
        pg_id = None
        for d, tag in gmsh.model.getPhysicalGroups():
            if d == dim and gmsh.model.getPhysicalName(d, tag) == physical_name:
                pg_id = tag
                break
        if pg_id is None:
            if strict:
                raise ValueError(f"未找到物理组: {physical_name}")
            return bm.array([], dtype=bm.int32)

        # 2) 收集该物理组下所有实体的 tags
        ent_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, pg_id)
        all_tags = []
        for etag in ent_tags:
            etypes, elemTagsList, _ = gmsh.model.mesh.getElements(dim, etag)
            for i, et in enumerate(etypes):
                if allowed_types and (et not in allowed_types):
                    continue
                if elemTagsList[i].size > 0:
                    all_tags.append(bm.asarray(elemTagsList[i], dtype=bm.int32))
        if all_tags:
            all_tags = bm.concatenate(all_tags, axis=0)
        else:
            all_tags = bm.empty((0,), dtype=bm.int32)

        if all_tags.size == 0:
            if strict:
                raise RuntimeError(f"物理组 {physical_name} 未发现元素")
            return bm.array([], dtype=bm.int32)

        # 3) （tag -> 全局索引）
        if not mapping:
            return bm.array([], dtype=bm.int32)
        map_keys = bm.array(list(mapping.keys()), dtype=bm.int32)
        map_vals = bm.array(list(mapping.values()), dtype=bm.int32)
        order = bm.argsort(map_keys)
        k_sorted = map_keys[order]
        v_sorted = map_vals[order]
        pos = bm.searchsorted(k_sorted, all_tags)
        mask = (pos < k_sorted.size) & (k_sorted[pos] == all_tags)
        idx = v_sorted[pos[mask]]
        if idx.size == 0:
            if strict:
                raise RuntimeError(f"物理组 {physical_name} 未匹配到全局索引")
            return bm.array([], dtype=bm.int32)

        return bm.unique(idx).astype(bm.int32)

    @variantmethod('tet')
    def init_mesh(self, return_bottom=False, 
                        return_inner_top=False, 
                        return_inner_head=False,
                        return_pml=False,
                        return_air=False,
                        return_plate=False,
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
            dict: Information about the PML volume if return_pml is True.
            dict: Information about the air volume if return_air is True.
            dict: Information about the plate volume if return_plate is True.
        """
        gmsh.initialize()
        try:
            gmsh.model.add("sphere_with_two_layer_plate")

            # 1) 几何：球、板、内层盒与凹槽，仅创建
            v_outer = gmsh.model.occ.addSphere(self.cx, self.cy, self.cz, self.R)
            v_inner = gmsh.model.occ.addSphere(self.cx, self.cy, self.cz, self.r)
            v_plate = gmsh.model.occ.addBox(self.x0, self.y0, self.z0, self.Lx, self.Ly, self.t)
            v_inbox = gmsh.model.occ.addBox(self.xi0, self.yi0, self.z0, self.lx_in, self.ly_in, self.t)
            notch_boxes = [gmsh.model.occ.addBox(hx0, hy0, self.z0, hsx, hsy, self.t)
                           for (hx0, hy0, hsx, hsy) in self.hole_defs]
            gmsh.model.occ.synchronize()

            # 2) 左右分割，仅用于分片
            xmid = 0.5 * (self.xi0 + self.xi1)
            if self.hole_defs and len(self.hole_defs) >= 2:
                holes2 = [{"x0": hx0, "x1": hx0 + hsx, "y0": hy0, "y1": hy0 + hsy, "sy": hsy}
                          for (hx0, hy0, hsx, hsy) in self.hole_defs]
                p = self._pick_notch_pair(holes2)
                if p: xmid = 0.5 * (p["xL"] + p["xR"])
            lw = max(0.0, xmid - self.xi0); rw = max(0.0, self.xi1 - xmid)
            v_left  = gmsh.model.occ.addBox(self.xi0, self.yi0, self.z0, lw, self.ly_in, self.t) if lw > 1e-12 else None
            v_right = gmsh.model.occ.addBox(xmid,     self.yi0, self.z0, rw, self.ly_in, self.t) if rw > 1e-12 else None
            gmsh.model.occ.synchronize()

            # 3) 一次 fragment：用工具把对象统一分片（避免多次 cut/intersect）
            objects = [(3, v_outer), (3, v_inner), (3, v_plate)]
            tools   = [(3, v_inbox)] + [(3, b) for b in notch_boxes]
            if v_left:  tools.append((3, v_left))
            if v_right: tools.append((3, v_right))
            ov_all, ov_map = gmsh.model.occ.fragment(objects, tools)
            gmsh.model.occ.synchronize()

            # 4) 通过映射分类（外/内/板/内盒/左右/凹槽）
            def map_set(i):
                return {tag for (d, tag) in ov_map[i] if d == 3} if (0 <= i < len(ov_map)) else set()

            idx_outer, idx_inner, idx_plate = 0, 1, 2
            idx_inbox = 3
            notch_start = 4
            notch_end   = notch_start + len(notch_boxes)
            idx_left_i  = notch_end if v_left else None
            idx_right_i = (notch_end + (1 if v_left else 0)) if v_right else None

            s_outer = map_set(idx_outer)
            s_inner = map_set(idx_inner)
            s_plate = map_set(idx_plate)
            s_inbox = map_set(idx_inbox)
            s_left  = map_set(idx_left_i)
            s_right = map_set(idx_right_i)
            s_notch = set()
            for i in range(notch_start, notch_end):
                s_notch |= map_set(i)

            # # 几何集合
            plate_all = set(s_plate)                                # 整块板（未扣凹槽）
            plate_inner_parts = sorted((s_plate & s_inbox) - s_notch)  # 内板 = (板 ∩ 内盒) − 凹槽
            plate_shell_parts = sorted((s_plate - s_inbox) | s_notch)  # 外板 = 板 − 内盒 + 凹槽

            v_shell_final = sorted(s_outer - s_inner)                # 壳 = 外 − 内
            v_core_parts  = sorted(s_inner - s_plate)                # 核 = 内 − 整块板（不因凹槽改变外板）

            # 5) 清理：仅保留 壳/核/板 这些碎体，避免工具碎体被网格化
            keep = set(v_shell_final) | set(v_core_parts) | set(plate_all)
            all3 = [t for (d, t) in gmsh.model.getEntities(3) if d == 3]
            drop = [t for t in all3 if t not in keep]
            if drop:
                gmsh.model.occ.remove([(3, t) for t in drop], recursive=True)
                gmsh.model.occ.synchronize()

            # 4) 物理分组
            if plate_inner_parts:
                pg_pin = gmsh.model.addPhysicalGroup(3, list(set(plate_inner_parts))); 
                gmsh.model.setPhysicalName(3, pg_pin, "VOL_PLATE_INNER")
            if plate_shell_parts:
                pg_psh = gmsh.model.addPhysicalGroup(3, list(set(plate_shell_parts))); 
                gmsh.model.setPhysicalName(3, pg_psh, "VOL_PLATE_SHELL")
            if plate_inner_parts or plate_shell_parts:
                pg_plate = gmsh.model.addPhysicalGroup(3, list(set(plate_inner_parts + plate_shell_parts))); 
                gmsh.model.setPhysicalName(3, pg_plate, "VOL_PLATE_WHOLE")
            if v_core_parts:
                pg_core  = gmsh.model.addPhysicalGroup(3, list(set(v_core_parts)));  
                gmsh.model.setPhysicalName(3, pg_core,  "VOL_SPHERE_AIR")
            if v_shell_final:
                pg_shell = gmsh.model.addPhysicalGroup(3, list(set(v_shell_final))); 
                gmsh.model.setPhysicalName(3, pg_shell, "VOL_SPHERE_PML")

            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

            # 5) 面物理组：底面、内层上表面、桥头小面
            self._tag_plate_bottom(plate_inner_parts, plate_shell_parts)
            self._tag_plate_inner_top(plate_inner_parts)
            # self._tag_bridge_head(plate_inner_parts, plate_shell_parts)

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
            if plate_inner_parts:
                gmshfidsetNs(fid_inner_res, "VolumesList", list(set(plate_inner_parts)))

            fid_shell_const = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(fid_shell_const, "F", f"{float(self.h_plate_shell)}")
            fid_shell_res = gmsh.model.mesh.field.add("Restrict")
            gmshfidsetN(fid_shell_res, "IField", fid_shell_const)
            if plate_shell_parts:
                gmshfidsetNs(fid_shell_res, "VolumesList", list(set(plate_shell_parts)))

            fid_ball = gmsh.model.mesh.field.add("Ball")
            gmshfidsetN(fid_ball, "VIn", float(self.h_sphere_inner))
            gmshfidsetN(fid_ball, "VOut", float(self.h_sphere_shell))
            gmshfidsetN(fid_ball, "XCenter", float(self.cx))
            gmshfidsetN(fid_ball, "YCenter", float(self.cy))
            gmshfidsetN(fid_ball, "ZCenter", float(self.cz))
            gmshfidsetN(fid_ball, "Radius", float(self.r) + 1e-12)

            f_list = [fid_ball]
            if plate_inner_parts: f_list.append(fid_inner_res)
            if plate_shell_parts: f_list.append(fid_shell_res)
            
            if self.hole_defs and len(self.hole_defs) >= 2:
                holes2 = [{"x0": hx0, "x1": hx0 + hsx, "y0": hy0, "y1": hy0 + hsy, "sy": hsy}
                          for (hx0, hy0, hsx, hsy) in self.hole_defs]
                p_pair = self._pick_notch_pair(holes2)
                if p_pair:
                    xL, xR, gap = p_pair["xL"], p_pair["xR"], p_pair["gap"]
                    yi0 = self.yi0
                    # 默认局部尺寸：若未提供 h_bridge_head，基于内板尺寸再缩小
                    _h_loc = float(self.h_bridge_head) if self.h_bridge_head is not None else 0.5 * float(self.h_plate_inner)
                    padX = max(_h_loc, 0.05 * gap)
                    padY = _h_loc
                    padZ = 0.2 * self.t
                    fid_box = gmsh.model.mesh.field.add("Box")
                    gmsh.model.mesh.field.setNumber(fid_box, "VIn", _h_loc)
                    # 外部给一个较大值，不影响其他 Min 场（仍被最小化）
                    gmsh.model.mesh.field.setNumber(fid_box, "VOut", max(self.h_plate_shell, self.h_sphere_shell))
                    gmsh.model.mesh.field.setNumber(fid_box, "XMin", xL - padX)
                    gmsh.model.mesh.field.setNumber(fid_box, "XMax", xR + padX)
                    gmsh.model.mesh.field.setNumber(fid_box, "YMin", yi0 - padY)
                    gmsh.model.mesh.field.setNumber(fid_box, "YMax", yi0 + padY)
                    gmsh.model.mesh.field.setNumber(fid_box, "ZMin", self.z0 - padZ)
                    gmsh.model.mesh.field.setNumber(fid_box, "ZMax", self.z0 + self.t + padZ)
                    f_list.append(fid_box)
            
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
            face = mesh.face

            cell_mapping = self._mapping_entity_to_ids(cell, nodetags_map)
            face_mapping = self._mapping_entity_to_ids(face ,nodetags_map)

            # 8) 可选表面数据
            info = {}
            ex_suf = self._extract_surface_by_name
            ex_vol = self._extract_volume_by_name

            if return_bottom:
                info['plate_bottom'] = ex_suf("SURF_PLATE_BOTTOM", face_mapping, strict=False)
            if return_inner_top:
                info['plate_inner_top'] = ex_suf("SURF_PLATE_INNER_TOP", face_mapping, strict=False)
            if return_inner_head:
                info['bridge_head_faces'] = self._find_bridge_head_faces_on_mesh(mesh)
            if return_inner_head:
                bh_faces = info.get('bridge_head_faces')
                if bh_faces is None:
                    bh_faces = self._find_bridge_head_faces_on_mesh(mesh)
                info['bridge_head_edges'] = self._find_bridge_head_center_edges_on_mesh(mesh, face_idx=bh_faces)
            if return_pml:
                info['pml'] = ex_vol("VOL_SPHERE_PML", cell_mapping, strict=False)
            if return_air:
                info['air'] = ex_vol("VOL_SPHERE_AIR", cell_mapping, strict=False)
            if return_plate:
                info['plate_whole'] = ex_vol("VOL_PLATE_WHOLE", cell_mapping, strict=False)

            if gmsh_show_fltk:
                gmsh.fltk.run()
            return mesh, info

        finally:
            gmsh.finalize()