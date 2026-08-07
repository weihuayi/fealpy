import gmsh
import trimesh
from fealpy.backend import backend_manager as bm
from scipy.spatial import cKDTree

class IntracranialAneurysm3dMesher:
    def __init__(self):
        pass

    def generate_mesh(self):
        # ============================================================
        # 参数区
        # ============================================================

        closed_surface_file = "closed_surface.stl"
        wall_surface_file = "wall.stl"

        output_vtk = "vessel.vtk"

        # test1: 0.4, 0.2, 0.8, 60w
        # test2: 0.3, 0.15, 1.0, 92w
        # 

        h_global = 0.4e-3    # 全局最大网格尺寸，mm
        h_wall = 0.08e-3      # wall 附近最大网格尺寸，mm

        # wall 附近加密影响距离
        # 含义：距离 wall 小于 d_wall 的区域，从 0.2 mm 逐渐过渡到 0.4 mm
        d_wall = 1.5e-3       # mm，可先取 0.6~1.0 mm

        eps = 1e-5         # 判断 closed_surface 顶点是否属于 wall 的距离容差


        # ============================================================
        # 0. 读取 closed surface 和 wall surface
        # ============================================================

        closed_mesh = trimesh.load(closed_surface_file, force="mesh")
        open_mesh = trimesh.load(wall_surface_file, force="mesh")

        V = bm.asarray(closed_mesh.vertices)
        F = bm.asarray(closed_mesh.faces)

        print("closed vertices:", V.shape)
        print("closed faces   :", F.shape)


        # ============================================================
        # 1. 识别 wall faces
        # ============================================================

        tree = cKDTree(bm.asarray(open_mesh.vertices))
        dist, _ = tree.query(V, k=1)

        closed_vertex_is_wall = dist < eps

        wall_mask = bm.all(closed_vertex_is_wall[F], axis=1)
        cap_mask = ~wall_mask

        print("wall faces:", wall_mask.sum())
        print("cap faces :", cap_mask.sum())


        # ============================================================
        # 2. 拆 cap 连通分量
        # ============================================================

        cap_mesh = trimesh.Trimesh(
            vertices=closed_mesh.vertices.copy(),
            faces=closed_mesh.faces[cap_mask],
            process=False,
        )

        components = cap_mesh.split(only_watertight=False)
        components = sorted(components, key=lambda m: m.area, reverse=True)

        print("cap components:", len(components))

        if len(components) != 6:
            print("WARNING: cap components is not 6. Current number =", len(components))


        # ============================================================
        # 3. 在 Gmsh 中创建多个 discrete surfaces
        # ============================================================

        gmsh.initialize()
        gmsh.model.add("vessel_direct_grouped_refined")

        triangle_type = 2  # 3-node triangle

        surface_tags = {}
        next_node_tag = 1
        next_elem_tag = 1


        def add_discrete_surface(name, vertices, faces, surf_tag):
            """
            把一组三角形直接作为一个 Gmsh discrete surface。
            注意：这里每个 surface 先独立建节点，
            后面 removeDuplicateNodes 合并重合节点。
            """
            nonlocal next_node_tag, next_elem_tag

            gmsh.model.addDiscreteEntity(2, surf_tag)

            nnode = len(vertices)
            node_tags = bm.arange(next_node_tag, next_node_tag + nnode, dtype=bm.int64)
            next_node_tag += nnode

            coords = vertices.reshape(-1).astype(float)

            gmsh.model.mesh.addNodes(
                2,
                surf_tag,
                node_tags.tolist(),
                coords.tolist(),
            )

            nelem = len(faces)
            elem_tags = bm.arange(next_elem_tag, next_elem_tag + nelem, dtype=bm.int64)
            next_elem_tag += nelem

            elem_node_tags = node_tags[faces].reshape(-1)

            gmsh.model.mesh.addElementsByType(
                surf_tag,
                triangle_type,
                elem_tags.tolist(),
                elem_node_tags.tolist(),
            )

            surface_tags[name] = surf_tag
            print(f"{name}: surface tag = {surf_tag}, faces = {nelem}")


        # wall surface
        add_discrete_surface(
            "wall",
            V,
            F[wall_mask],
            surf_tag=1,
        )

        # cap surfaces
        for i, comp in enumerate(components, start=1):
            add_discrete_surface(
                f"cap_{i}",
                bm.asarray(comp.vertices),
                bm.asarray(comp.faces),
                surf_tag=i + 1,
            )


        # ============================================================
        # 4. 合并重复节点，建立拓扑
        # ============================================================

        gmsh.model.mesh.removeDuplicateNodes([])
        gmsh.model.mesh.removeDuplicateElements([])
        gmsh.model.mesh.renumberNodes()
        gmsh.model.mesh.renumberElements()

        gmsh.model.mesh.createTopology()


        # ============================================================
        # 5. 创建 volume
        # ============================================================

        all_surfaces = list(surface_tags.values())

        sl = gmsh.model.geo.addSurfaceLoop(all_surfaces)
        vol = gmsh.model.geo.addVolume([sl])
        gmsh.model.geo.synchronize()


        # ============================================================
        # 6. 设置 Physical Groups
        # ============================================================

        name_to_phys = {
            "wall": 1,
            "cap_1": 2,
            "cap_2": 3,
            "cap_3": 4,
            "cap_4": 5,
            "cap_5": 6,
            "cap_6": 7,
        }

        for name, phys_id in name_to_phys.items():
            if name in surface_tags:
                tag = surface_tags[name]
                gmsh.model.addPhysicalGroup(2, [tag], phys_id)
                gmsh.model.setPhysicalName(2, phys_id, name)

        gmsh.model.addPhysicalGroup(3, [vol], 1)
        gmsh.model.setPhysicalName(3, 1, "fluid")


        # ============================================================
        # 7. 设置全局尺寸 + wall 附近局部加密
        # ============================================================

        gmsh.option.setNumber("Mesh.MeshSizeMin", h_wall)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h_global)

        # 关闭自动从边界曲率/点继承尺寸，避免尺寸场被干扰
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        # 取得 wall surface 上的节点
        wall_tag = surface_tags["wall"]

        wall_nodes, wall_coords, _ = gmsh.model.mesh.getNodes(
            2,
            wall_tag,
            includeBoundary=True
        )

        wall_nodes = [int(t) for t in wall_nodes]

        print("wall nodes for distance field:", len(wall_nodes))

        # Distance field：计算体内点到 wall nodes 的距离
        f_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist, "NodesList", wall_nodes)

        # Threshold field：
        # d = 0              -> h = h_wall
        # d >= d_wall       -> h = h_global
        # 0 < d < d_wall    -> 从 h_wall 过渡到 h_global
        f_th = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_th, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_th, "SizeMin", h_wall)
        gmsh.model.mesh.field.setNumber(f_th, "SizeMax", h_global)
        gmsh.model.mesh.field.setNumber(f_th, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(f_th, "DistMax", d_wall)

        gmsh.model.mesh.field.setAsBackgroundMesh(f_th)


        # ============================================================
        # 8. 生成体网格
        # ============================================================

        # 只填充空的 volume，不重划已有 discrete surface mesh
        gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 0)

        # 可选：提高 3D 网格优化
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.generate(3)

        # 基础优化
        gmsh.model.mesh.optimize("")

        # Netgen 优化，通常对四面体质量更有效
        gmsh.model.mesh.optimize("Netgen")

        # 可选：再做一轮基础优化
        gmsh.model.mesh.optimize("")

        node_tags, coords, _ = gmsh.model.mesh.getNodes()

        node_tags = bm.asarray(node_tags, dtype=bm.int64)
        node = bm.asarray(coords, dtype=bm.float64).reshape(-1, 3)

        # Gmsh 的节点编号 node_tags 通常从 1 开始，不适合直接当 Python 下标
        # 所以要建立：gmsh节点编号 -> numpy数组下标 的映射
        tag_to_index = {tag: i for i, tag in enumerate(node_tags)}

        # 2. 提取四面体单元
        # Gmsh 中 4-node tetrahedron 的 element type 是 4
        tet_type = 4

        elem_tags, elem_node_tags = gmsh.model.mesh.getElementsByType(tet_type)

        elem_node_tags = bm.asarray(elem_node_tags, dtype=bm.int64).reshape(-1, 4)

        # 3. 把 Gmsh 节点编号转成 0-based 的 cell
        cell = bm.array(
            [[tag_to_index[t] for t in tet] for tet in elem_node_tags],
            dtype=bm.int64
        )
        boundary_faces = self.extract_physical_boundary_faces(tag_to_index)

        for name, faces in boundary_faces.items():
            print(name, faces.shape)
        from fealpy.mesh import TetrahedronMesh
        mesh = TetrahedronMesh(node=node, cell=cell)
        mesh.boundary_faces = boundary_faces
        mesh.inlet_face_index = self.boundary_faces_to_face_index(mesh, boundary_faces["cap_1"])

        # 9. 输出

        gmsh.write(output_vtk)

        gmsh.finalize()
        return mesh
        
    def extract_physical_boundary_faces(self, tag_to_index):
        """
        提取 Gmsh 中二维 Physical Groups 的边界三角形。
        
        返回:
            boundary_faces: dict
                key   : physical name, 例如 "wall", "cap_1"
                value : face array, shape = (NF, 3), 0-based node index
        """
        import numpy as np

        boundary_faces = {}

        physical_groups = gmsh.model.getPhysicalGroups(dim=2)

        for dim, phys_tag in physical_groups:
            name = gmsh.model.getPhysicalName(dim, phys_tag)

            entity_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag)

            face_list = []

            for entity_tag in entity_tags:
                elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(
                    dim,
                    entity_tag
                )

                for etype, enodes in zip(elem_types, elem_node_tags):
                    # 2 表示 3-node triangle
                    if etype == 2:
                        tri_node_tags = bm.asarray(enodes, dtype=bm.int64).reshape(-1, 3)

                        tri = bm.array(
                            [[tag_to_index[t] for t in tri_nodes]
                            for tri_nodes in tri_node_tags],
                            dtype=bm.int64
                        )

                        face_list.append(tri)

            if len(face_list) > 0:
                boundary_faces[name] = bm.concatenate(face_list)

        return boundary_faces
    
    def boundary_faces_to_face_index(self, mesh, boundary_faces):
        """
        把 boundary_faces 中的三角形节点编号转换成 FEALPy mesh.face 的全局 face index。

        Parameters
        ----------
        mesh : TetrahedronMesh
        boundary_faces : ndarray, shape = (NF, 3)

        Returns
        -------
        face_index : ndarray, shape = (NF,)
        """

        # FEALPy 全局 face
        mesh_faces = bm.asarray(mesh.entity('face'))

        # 你的边界 face
        bfaces = bm.asarray(boundary_faces)

        # 为了消除节点顺序影响，对每个三角形的节点编号排序
        mesh_faces_sorted = bm.sort(mesh_faces, axis=1)
        bfaces_sorted = bm.sort(bfaces, axis=1)

        # 建立：排序后的节点三元组 -> face index
        face_dict = {
            tuple(f): i for i, f in enumerate(mesh_faces_sorted)
        }

        face_index = bm.array(
            [face_dict[tuple(f)] for f in bfaces_sorted],
            dtype=bm.int64
        )

        return face_index
            