import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch import relu
from mpl_toolkits.mplot3d import Axes3D
import gmsh
import pickle

from fealpy.mesh import TriangleMesh, TetrahedronMesh
from fealpy.backend import backend_manager as bm

bm.set_backend('pytorch')


def draw_mesh_and_face_normal(mesh, face_normal):
    bd_face_idx = mesh.boundary_face_index()
    face_centers = mesh.entity_barycenter('face', bd_face_idx)

    # 创建3D绘图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制法向量箭头（颜色编码方向）
    quiver = ax.quiver(
        face_centers[:, 0], face_centers[:, 1], face_centers[:, 2],  # 起点
        face_normal[:, 0], face_normal[:, 1], face_normal[:, 2],  # 方向
        length=0.2,  # 箭头长度
        color=plt.cm.jet(face_normal[:, 2]),  # 用Z分量映射颜色
        arrow_length_ratio=0.5  # 箭头头部比例
    )
    origin_mesh.add_plot(ax)

    plt.show()

def get_origin_mesh():
    gmsh.initialize()
    gmsh.model.add("tet_mesh")

    mesh = TriangleMesh.from_unit_sphere_surface(0)
    nodes = mesh.node
    cells = mesh.cell

    # 1. 添加所有点，并记录 Gmsh 返回的点 tag
    point_tags = []
    for coord in nodes:
        tag = gmsh.model.geo.addPoint(coord[0], coord[1], coord[2])
        point_tags.append(tag)

    # 2. 对所有三角形面，构造曲线并去除重复（共享边只创建一次）
    curve_tags = {}  # 键为（起点, 终点）的无序元组，值为曲线 tag
    face_tags = []  # 保存所有面的 tag

    for cell in cells:
        # 对于每个三角形，取出对应的点 tag
        pts = [point_tags[idx] for idx in cell]
        lines = []
        for j in range(3):
            i1 = pts[j]
            i2 = pts[(j + 1) % 3]
            # 为了忽略方向，排序后作为键
            key = tuple(sorted((i1, i2)))
            if key in curve_tags:
                line_tag = curve_tags[key]
            else:
                line_tag = gmsh.model.geo.addLine(i1, i2)
                curve_tags[key] = line_tag
            lines.append(line_tag)
        # 构造曲线环（Curve Loop）和对应的平面
        loop = gmsh.model.geo.addCurveLoop(lines, reorient=True)
        face = gmsh.model.geo.addPlaneSurface([loop])
        face_tags.append(face)

    # 3. 用所有面构造封闭体
    surface_loop = gmsh.model.geo.addSurfaceLoop(face_tags)
    vol = gmsh.model.geo.addVolume([surface_loop])

    # 4. 同步几何模型，生成三维网格（四面体）
    gmsh.option.setNumber("Mesh.MeshSizeFactor", 2.5)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # 使用 Delaunay 四面体网格
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.optimize("Netgen")
    gmsh.model.mesh.generate(3)
    # 读取网格节点与单元
    # 获取所有节点信息
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_tags = bm.array(node_tags)
    node_coords = bm.array(node_coords)
    node = node_coords.reshape((-1, 3))
    # 节点编号映射
    nodetags_map = dict({int(j): i for i, j in enumerate(node_tags)})
    # 获取单元信息
    cell_type = 4
    cell_tags, cell_connectivity = gmsh.model.mesh.getElementsByType(cell_type)
    # 节点编号映射到单元
    evid = bm.array([nodetags_map[int(j)] for j in cell_connectivity])
    cell = evid.reshape((cell_tags.shape[-1], -1))

    # gmsh.fltk.run()
    # 5. 保存网格并退出
    # gmsh.write("icosahedron_tet.msh")
    gmsh.finalize()
    tet_mesh = TetrahedronMesh(node, cell)
    return tet_mesh

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # tet_mesh.add_plot(ax)
    # plt.show()

def closed_axis_projection(v):
    sign = bm.sign(v).reshape(-1, 3)
    abs_max_axis = bm.argmax(bm.abs(v), axis=-1).reshape(-1)
    proj = bm.zeros_like(v).reshape(-1, 3)
    row_idx = bm.arange(proj.shape[0])
    proj[row_idx, abs_max_axis] = sign[row_idx, abs_max_axis]

    return proj.reshape(v.shape)

def compute_gaussian_normals(face_centers, areas, face_normal, sigma):
    t1 = -bm.linalg.norm(face_centers[:, None, :] - face_centers[None, ...], axis=-1)**2/2/sigma**2
    t2 = bm.exp(t1)
    gaussian_normals = bm.einsum('j,ij,jd->id', areas, t2, face_normal)
    gaussian_normals = gaussian_normals / bm.linalg.norm(gaussian_normals, axis=-1, keepdims=True)
    # diff = gaussian_normals - face_normal

    return gaussian_normals

def compute_smooth_normal_energy(gauss_normals, origin_normals):
    diff = origin_normals - closed_axis_projection(gauss_normals)
    energy = bm.linalg.norm(diff, axis=-1)**2
    return energy

def compute_AMIPS_energy(node_new, node_old, cell, s=1, alpha=0.5):
    def deformation_matrix(tet_node, tet_node_new):
        p_def = tet_node_new[:, 0, :][..., None]  # 变形后的顶点 p，形状 (NC, 3)
        def_edges = bm.concat([
            p_def - tet_node_new[:, 1, :][..., None],  # v_p - v_q
            p_def - tet_node_new[:, 2, :][..., None],  # v_p - v_r
            p_def - tet_node_new[:, 3, :][..., None]  # v_p - v_s
        ], axis=-1)  # 形状 (NC, 3, 3)

        p_orig = tet_node[:, 0, :][..., None]  # 变形前的顶点 p，形状 (NC, 3)
        orig_edges = bm.concat([
            p_orig - tet_node[:, 1, :][..., None],  # v_p0 - v_q0
            p_orig - tet_node[:, 2, :][..., None],  # v_p0 - v_r0
            p_orig - tet_node[:, 3, :][..., None]  # v_p0 - v_s0
        ], axis=-1)  # 形状 (NC, 3, 3)

        # 批量求逆矩阵
        inv_orig_edges = bm.linalg.inv(orig_edges)  # 形状 (NC, 3, 3)

        # 矩阵乘法：A = def_edges @ inv_orig_edges
        A = bm.einsum('nij,njk->nik', def_edges, inv_orig_edges)

        return A

    tet_node = node_old[cell]
    tet_node_new = node_new[cell]
    A = deformation_matrix(tet_node, tet_node_new)
    A_inv = bm.linalg.inv(A)

    delta_conf = bm.einsum('nij->n', A**2)*bm.einsum('nij->n', A_inv**2)/8
    delta_vol = bm.linalg.det(A)*bm.linalg.det(A_inv)/2

    e_iso = bm.exp(s*(alpha*delta_conf + (1-alpha)*delta_vol))

    return e_iso

def phi(x):
    nx = x[..., 0]**2
    ny = x[..., 1]**2
    nz = x[..., 2]**2
    phi = nx*ny + ny*nz + nz*nx

    return phi

def compute_e_o(rotate_matrix, face_normal):
    rotate_normal = bm.einsum('ij,...j->...i', rotate_matrix, face_normal)
    e_o = bm.sum(phi(rotate_normal))

    return e_o

def compute_e_ns(rotate_matrix, node_new, node_old, cell,
                 face_centers, areas, face_normal, face2cell,
                 sigma=0.1, s=1, alpha=0.5):
    face_normal = bm.einsum('ij,...j->...i', rotate_matrix, face_normal)
    gaussian_normals = compute_gaussian_normals(face_centers, areas, face_normal, sigma)
    # gaussian_normals = bm.einsum('ij,...j->...i', rotate_matrix, gaussian_normals)
    # face_normal = bm.einsum('ij,...j->...i', rotate_matrix, face_normal)
    e_s_elem = compute_smooth_normal_energy(gaussian_normals, face_normal)
    e_iso_elem = compute_AMIPS_energy(node_new, node_old, cell, s, alpha)

    mu_min = bm.array(1e3)
    mu_max = bm.array(1e16)
    mu = bm.minimum(mu_max, bm.maximum(mu_min, e_iso_elem[face2cell]/e_s_elem))
    # mu = bm.ones(e_s_elem.shape[0], dtype=bm.float64)

    e_s = bm.einsum('i,i->', mu, e_s_elem)
    e_iso = bm.sum(e_iso_elem)
    e_ns = e_s + e_iso
    return e_ns

def update(origin_mesh:TetrahedronMesh, sigma=0.1, s=1, alpha=0.5):
    origin_node = origin_mesh.node
    cell = origin_mesh.cell
    new_node = bm.zeros_like(origin_node, dtype=bm.float64, requires_grad=True)
    new_node.data = origin_node.data
    old_node = origin_node
    # TODO: 旋转矩阵是否需要改成每个面独立，而不是当前的所有面统一旋转矩阵
    rotate_matrix = bm.eye(3, dtype=bm.float64, requires_grad=True)

    lr = 0.001  # 学习率
    max_num_epochs = 10000  # 迭代次数
    error = 1e-3
    pre_energy = 1e16

    # 创建优化器（Adam 优化器）
    optimizer_node = optim.Adam([new_node], lr=lr)
    optimizer_rotate = optim.Adam([rotate_matrix], lr=lr)
    # 显式迭代优化过程
    for step in range(max_num_epochs):
        # 更新网格
        mesh = TetrahedronMesh(new_node, cell)
        bd_face_idx = mesh.boundary_face_index()
        face_centers = mesh.entity_barycenter('face', bd_face_idx)
        areas = mesh.entity_measure('face', bd_face_idx)
        face2cell = mesh.face2cell[bd_face_idx, 0]

        # --- 优化 rotate_matrix ---
        face_normal = mesh.face_unit_normal(bd_face_idx)
        # face_normal = mesh.face_unit_normal(bd_face_idx).detach()
        optimizer_rotate.zero_grad()
        energy_o = compute_e_o(rotate_matrix, face_normal)
        energy_o.backward(retain_graph=True)
        optimizer_rotate.step()
        # --- 优化 node ---
        # face_normal = mesh.face_unit_normal(bd_face_idx)
        optimizer_node.zero_grad()
        energy_ns = compute_e_ns(rotate_matrix, new_node,
                                 old_node, cell, face_centers, areas, face_normal, face2cell,
                                 sigma, s, alpha)
        energy_ns.backward()
        old_node = new_node.detach().clone()
        optimizer_node.step()

        # 每隔50步输出
        if (step + 1) % 50 == 0:
            print(f"Step [{step + 1}], Energy: {energy_ns.item():.4f}")

        if (step==max_num_epochs-1) or (bm.linalg.norm(new_node - old_node) < error and (energy_ns > pre_energy)):
            print(face_normal)
            t = bm.einsum('ij,nj->ni', rotate_matrix, face_normal)
            print(t / bm.linalg.norm(t, axis=1).reshape(-1, 1))
            break
        else:
            pre_energy = energy_ns

    optimized_mesh = TetrahedronMesh(new_node, cell)
    # pickle.dump(optimized_mesh, open("optimized_mesh.pkl", "wb"))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    optimized_mesh.add_plot(ax)
    ax.set_title("Optimized Mesh")
    plt.show()


if __name__ == "__main__":
    # origin_mesh = get_origin_mesh()
    # pickle.dump(origin_mesh, open("origin_mesh_torch.pkl", "wb"))
    origin_mesh = pickle.load(open("origin_mesh_torch.pkl", "rb"))
    # origin_mesh = TetrahedronMesh.from_box(nx=2, ny=2, nz=2)
    # node = origin_mesh.node
    # cell = origin_mesh.cell
    # face = origin_mesh.face
    # bd_face_idx = origin_mesh.boundary_face_index()
    # bd_face2cell = origin_mesh.face_to_cell(bd_face_idx)[:, 0]
    # bd_face = face[bd_face_idx]
    #
    # bd_face_node = node[bd_face]

    update(origin_mesh, sigma=0.1, s=1, alpha=0.5)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # origin_mesh.add_plot(ax)
    # ax.set_title("Origin Mesh")
    # plt.show()














