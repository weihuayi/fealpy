from typing import Sequence
from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.mesh import TriangleMesh, QuadrangleMesh
import numpy as np
import gmsh
    
class AnnulusMesher:
    """
    使用 gmsh 构造 2D 圆环或“扇形环”几何并三角剖分，导出 FEALPy TriangleMesh。
    """
    def init_mesh(
        self,
        R: float = 1.0,
        r: float = 0.3,
        theta0_deg: float = 0.0,
        theta1_deg: float = 360.0,
        center=(0.0, 0.0),
        h: float = 0.05, ) -> TriangleMesh:
        assert R > 0 and r > 0 and R > r, "需要 R > r > 0"
        cx, cy = center
        gmsh.initialize()
        gmsh.model.add("Annulus2D")
        # 归一化角度，判断是否满圈
        t0 = float(theta0_deg) % 360.0
        delta = (float(theta1_deg) - t0) % 360.0
        full_circle = np.isclose(delta, 0.0) or np.isclose(delta, 360.0)
        a0 = np.deg2rad(t0)
        a1 = a0 + (0 if full_circle else np.deg2rad(delta))
        # 工具函数
        def add_pt(x, y):
            return gmsh.model.occ.addPoint(x, y, 0.0, h)
        def circ_pt(radius, ang):
            return add_pt(cx + radius * np.cos(ang), cy + radius * np.sin(ang))
        cpt = add_pt(cx, cy)  # 圆心点（用于 addCircleArc）
        if full_circle:
            # 用 4 个 90° 圆弧构造外、内闭合边界，再用平面曲面+内环洞生成圆环
            # 外圈四点（0,90,180,270）
            po0 = circ_pt(R, 0.0)
            po1 = circ_pt(R, 0.5 * np.pi)
            po2 = circ_pt(R, np.pi)
            po3 = circ_pt(R, 1.5 * np.pi)
            ao0 = gmsh.model.occ.addCircleArc(po0, cpt, po1)
            ao1 = gmsh.model.occ.addCircleArc(po1, cpt, po2)
            ao2 = gmsh.model.occ.addCircleArc(po2, cpt, po3)
            ao3 = gmsh.model.occ.addCircleArc(po3, cpt, po0)
            outer_loop = gmsh.model.occ.addCurveLoop([ao0, ao1, ao2, ao3])
            # 内圈四点（同序），曲线环在面中作为“洞”需要反向
            pi0 = circ_pt(r, 0.0)
            pi1 = circ_pt(r, 0.5 * np.pi)
            pi2 = circ_pt(r, np.pi)
            pi3 = circ_pt(r, 1.5 * np.pi)
            ai0 = gmsh.model.occ.addCircleArc(pi0, cpt, pi1)
            ai1 = gmsh.model.occ.addCircleArc(pi1, cpt, pi2)
            ai2 = gmsh.model.occ.addCircleArc(pi2, cpt, pi3)
            ai3 = gmsh.model.occ.addCircleArc(pi3, cpt, pi0)
            inner_loop = gmsh.model.occ.addCurveLoop([-ai0, -ai1, -ai2, -ai3])  # 反向作为洞
            surf = gmsh.model.occ.addPlaneSurface([outer_loop, inner_loop])
        else:
            # 扇形环：外弧、内弧、两条径向线围成一个曲线环
            po0 = circ_pt(R, a0)
            po1 = circ_pt(R, a1)
            pi0 = circ_pt(r, a0)
            pi1 = circ_pt(r, a1)
            arc_outer = gmsh.model.occ.addCircleArc(po0, cpt, po1)
            arc_inner_forward = gmsh.model.occ.addCircleArc(pi0, cpt, pi1)
            # 径向线（方向选取配合环的正向闭合）
            line_end = gmsh.model.occ.addLine(po1, pi1)
            line_start = gmsh.model.occ.addLine(pi0, po0)
            loop = gmsh.model.occ.addCurveLoop([arc_outer, line_end, -arc_inner_forward, -line_start])
            surf = gmsh.model.occ.addPlaneSurface([loop])
        gmsh.model.occ.synchronize()
        # 网格尺寸
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(h))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(h))
        # 2D 三角网格
        gmsh.model.mesh.generate(2)
        # 获取节点信息
        node_coords = gmsh.model.mesh.getNodes()[1]
        node = np.array(node_coords, dtype=np.float64).reshape(-1, 3)
        # 获取三角形单元信息
        triangle_type = 2  
        triangle_tags, triangle_connectivity = gmsh.model.mesh.getElementsByType(triangle_type)
        cell = np.array(triangle_connectivity, dtype=np.int_).reshape(-1, 3) -1
        # 获得正确的节点标签
        NN = len(node)
        isValidNode = np.zeros(NN, dtype=np.bool_)
        isValidNode[cell] = True
        # 去除未三角化的点
        node = node[isValidNode,:2]
        idxMap = np.zeros(NN, dtype=cell.dtype)
        idxMap[isValidNode] = range(isValidNode.sum())
        cell = idxMap[cell]
        print(f"Number of nodes: {node.shape[0]}")
        print(f"Number of cells: {cell.shape[0]}")
        gmsh.finalize()
        node = bm.array(node)
        cell = bm.array(cell)
        mesh = TriangleMesh(node, cell)
        return mesh