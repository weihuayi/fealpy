import gmsh
import numpy as np

from fealpy.mesh import EdgeMesh
from fealpy.backend import bm
import matplotlib.pyplot as plt


class TrussTower:
    def build_truss_3d_zbar(n_panel=8, Lz=20.0, Wx=2.0, Wy=1.0, lc=0.1,
                            ne_per_bar=1, face_diag=True, save_msh=None):
        """
        生成 3D z 轴方向的长条状桁架（线单元）。
        
        参数:
            n_panel    : z 方向面板数(>=1)
            Lz         : 总长度(沿 z)
            Wx, Wy     : 截面矩形尺寸(x 宽、y 高)
            lc         : 几何点特征长度
            ne_per_bar : 每根杆件沿长度等分段数(>=1)
            face_diag  : 是否在四个侧面加面内对角加劲
            save_msh   : 可选 *.msh 保存
        返回:
            node (N,3) : 节点坐标
            cell (E,2) : 线单元端点(0-based)
        """
        assert n_panel >= 1 and ne_per_bar >= 1
        gmsh.initialize()
        gmsh.model.add("zbar_truss")
        geo = gmsh.model.geo

        # 0) z 方向分站
        zs = bm.linspace(0.0, Lz, n_panel + 1)
        hx, hy = Wx/2.0, Wy/2.0

        # 1) 每个 z 截面放置四角点：顺序 [(-x,-y), (x,-y), (x,y), (-x,y)]
        planes = []  # [[p0,p1,p2,p3], ...] 每层四个点的 tag
        for z in zs:
            p0 = geo.addPoint(-hx, -hy, float(z), lc)
            p1 = geo.addPoint( hx, -hy, float(z), lc)
            p2 = geo.addPoint( hx,  hy, float(z), lc)
            p3 = geo.addPoint(-hx,  hy, float(z), lc)
            planes.append([p0, p1, p2, p3])

        lines = []

        def add_line(a, b):
            lines.append(geo.addLine(a, b))

        # 2) 纵向杆件（沿 z）：四个角分别连通相邻截面
        for j in range(4):
            for k in range(n_panel):
                add_line(planes[k][j], planes[k+1][j])

        # 3) 每个截面四边（环向连杆）
        for k in range(n_panel + 1):
            add_line(planes[k][0], planes[k][1])
            add_line(planes[k][1], planes[k][2])
            add_line(planes[k][2], planes[k][3])
            add_line(planes[k][3], planes[k][0])

        # 4) 侧面对角加劲（可选），在四个侧面 (y=±hy, x=±hx) 上交替布置
        if face_diag:
            # y = -hy 面：角点索引 (0,1)
            for k in range(n_panel):
                if k % 2 == 0:
                    add_line(planes[k][0], planes[k+1][1])
                else:
                    add_line(planes[k][1], planes[k+1][0])
            # y = +hy 面：角点索引 (3,2)
            for k in range(n_panel):
                if k % 2 == 0:
                    add_line(planes[k][2], planes[k+1][3])
                else:
                    add_line(planes[k][3], planes[k+1][2])
            # x = -hx 面：角点索引 (0,3)
            for k in range(n_panel):
                if k % 2 == 0:
                    add_line(planes[k][0], planes[k+1][3])
                else:
                    add_line(planes[k][3], planes[k+1][0])
            # x = +hx 面：角点索引 (1,2)
            for k in range(n_panel):
                if k % 2 == 0:
                    add_line(planes[k][2], planes[k+1][1])
                else:
                    add_line(planes[k][1], planes[k+1][2])

        for k in range(n_panel+1):
            if k % 2 == 0:
                add_line(planes[k][0], planes[k][2])
            else:
                add_line(planes[k][1], planes[k][3])
        # 5) 同步 & 等分每根线
        geo.synchronize()
        for ltag in lines:
            gmsh.model.mesh.setTransfiniteCurve(ltag, ne_per_bar + 1)

        # 6) 物理分组（可选）：全部线
        pg_all = gmsh.model.addPhysicalGroup(1, lines)
        gmsh.model.setPhysicalName(1, pg_all, "Bars3D")

        # 7) 生成 1D 网格
        gmsh.model.mesh.generate(1)
        if save_msh:
            gmsh.write(save_msh)

        # 8) 提取节点与线单元（type=1: 2-node line）
        nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
        nodeTags = nodeTags.astype(bm.int64)
        coords = nodeCoords.reshape(-1, 3)
        node = coords[:, :3]
        types, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=1)
        line_cells = None
        for t, conn in zip(types, elemNodeTags):
            if t == 1:
                line_cells = conn.reshape(-1, 2).astype(bm.int64)
                break
        if line_cells is None:
            gmsh.finalize()
            raise RuntimeError("未找到 type=1 的线单元。")

        tag_to_id = {tag: i for i, tag in enumerate(nodeTags)}
        cell = np.vectorize(tag_to_id.get)(line_cells)

        gmsh.finalize()
        return node, cell

    def quick_plot_3d(node, cell, title="3D z-bar truss"):
        fig = plt.figure(figsize=(3, 8))
        ax = fig.add_subplot(111, projection='3d')
        for e in cell:
            xyz = node[e]
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'k-')
        # ax.scatter(node[:, 0], node[:, 1], node[:, 2], s=5, c='r')
        ax.set_box_aspect((np.ptp(node[:,0]), np.ptp(node[:,1]), np.ptp(node[:,2]) + 1e-12))
        ax.set_title(title)
        ax.set_axis_off()
        plt.tight_layout(); plt.show()

    # if __name__ == "__main__":
    #     # 示例：z 向 8 面板，长 20，截面 2x1，每杆 2 段
    #     node, cell = build_truss_3d_zbar(n_panel=19, Lz=19.0, Wx=1.0, Wy=1.0,
    #                                     lc=0.1, ne_per_bar=2, face_diag=True, save_msh="truss_3d.msh")
    #     quick_plot_3d(node, cell, "3D z-bar truss")
        

node, cell = TrussTower.build_truss_3d_zbar(n_panel=19, Lz=19, Wx=0.45, Wy=0.40, lc=0.1, ne_per_bar=1)
mesh = EdgeMesh(node, cell)

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d') 
mesh.add_plot(axes)
plt.show()