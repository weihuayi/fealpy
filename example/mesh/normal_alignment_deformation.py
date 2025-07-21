from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh, TetrahedronMesh
import torch.optim as optim
from torch import relu
import matplotlib.pyplot as plt
import pickle


class MeshNormalAlignmentDeformation:
    # 六个轴方向
    axes = bm.tensor([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=bm.float64)

    def __init__(self, mesh):
        self.mesh: TetrahedronMesh = mesh
        # self.bd_mesh = self.get_bd_mesh()
        # self.face_target_normal = face_target_normal

    def get_bd_mesh(self) -> TriangleMesh:
        """从体网格获取边界网格"""
        volume_mesh = self.mesh
        bd_face_idx = volume_mesh.boundary_face_index()
        node = volume_mesh.node
        face = volume_mesh.face
        bd_node_idx = volume_mesh.boundary_node_index()
        bd_node = node[bd_node_idx]
        idx_map = bm.zeros(node.shape[0], dtype=bd_node_idx.dtype)
        idx_map[bd_node_idx] = bm.arange(bd_node_idx.shape[0], dtype=bd_node_idx.dtype)
        bd_cell = idx_map[face[bd_face_idx]]

        return TriangleMesh(bd_node, bd_cell)

    @staticmethod
    def phi(x):
        nx = x[..., 0] ** 2
        ny = x[..., 1] ** 2
        nz = x[..., 2] ** 2
        phi = nx * ny + ny * nz + nz * nx

        return phi

    def compute_AMIPS_energy(self, node_new, node_old, cell, s=1, alpha=0.5):
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

        delta_conf = bm.einsum('nij->n', A ** 2) * bm.einsum('nij->n', A_inv ** 2) / 8
        delta_vol = bm.linalg.det(A) * bm.linalg.det(A_inv) / 2

        e_iso = bm.exp(s * (alpha * delta_conf + (1 - alpha) * delta_vol))

        return e_iso


    def compute_e_na(self, rotate_matrix, node_new, node_old, cell, face_normal, gamma=1e3,
                     s=1, alpha=0.5):
        r_n = (rotate_matrix @ face_normal.T).T
        phi_n = self.phi(r_n)
        e_iso_elem = self.compute_AMIPS_energy(node_new, node_old, cell, s, alpha)

        face2cell = self.mesh.face2cell[self.mesh.boundary_face_index(), 0]
        omega_min = bm.array(1e3)
        omega_max = bm.array(1e16)
        omega = bm.minimum(omega_max, gamma*bm.maximum(omega_min, e_iso_elem[face2cell] / phi_n))

        e_a = bm.sum(omega.reshape(-1) * phi_n)
        e_iso = bm.sum(e_iso_elem)
        e_na = e_iso + e_a
        return e_na

    def compute_e_o(self, rotate_matrix, face_normal):
        rotate_normal = bm.einsum('ij,...j->...i', rotate_matrix, face_normal)
        e_o = bm.sum(self.phi(rotate_normal))
        return e_o

    def update_mesh(self):
        """更新网格"""
        origin_node = self.mesh.node
        cell = self.mesh.cell
        new_node = bm.zeros_like(origin_node, dtype=bm.float64, requires_grad=True)
        new_node.data = origin_node.data
        old_node = origin_node
        # TODO: 旋转矩阵是否需要改成每个面独立，而不是当前的所有面统一旋转矩阵
        rotate_matrix = bm.eye(3, dtype=bm.float64, requires_grad=True)

        lr = 0.001  # 学习率
        max_num_epochs = 10000  # 迭代次数
        error = 1e-5
        pre_energy = 1e16
        gamma = 1e3

        # 创建优化器（Adam 优化器）
        optimizer_node = optim.Adam([new_node], lr=lr)
        optimizer_rotate = optim.Adam([rotate_matrix], lr=lr)
        # 显式迭代优化过程
        for step in range(max_num_epochs):
            # 更新网格
            self.mesh = TetrahedronMesh(new_node, cell)
            bd_face_idx = self.mesh.boundary_face_index()

            # --- 优化 rotate_matrix ---
            face_normal = self.mesh.face_unit_normal(bd_face_idx)
            # face_normal = mesh.face_unit_normal(bd_face_idx).detach()
            optimizer_rotate.zero_grad()
            energy_o = self.compute_e_o(rotate_matrix, face_normal)
            energy_o.backward(retain_graph=True)
            optimizer_rotate.step()
            # --- 优化 node ---
            # face_normal = mesh.face_unit_normal(bd_face_idx)
            optimizer_node.zero_grad()
            energy_na = self.compute_e_na(rotate_matrix, new_node, old_node, cell, face_normal, gamma)
            energy_na.backward()
            old_node = new_node.detach().clone()
            optimizer_node.step()
            # gamma = gamma * 10

            # 每隔50步输出
            if (step + 1) % 50 == 0:
                print(f"Step [{step + 1}], Energy: {energy_na.item():.4f}")

            if (step == max_num_epochs - 1) or (
                    bm.linalg.norm(new_node - old_node) < error and (energy_na > pre_energy)):
                print(face_normal)
                t = bm.einsum('ij,nj->ni', rotate_matrix, face_normal)
                print(t/bm.linalg.norm(t, axis=1).reshape(-1, 1))
                break
            else:
                pre_energy = energy_na

        optimized_mesh = TetrahedronMesh(new_node, cell)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        optimized_mesh.add_plot(ax)
        ax.set_title("Optimized Mesh")
        plt.show()



if __name__ == "__main__":
    bm.set_backend('pytorch')
    origin_mesh = pickle.load(open("origin_mesh_torch.pkl", "rb"))

    mesh_deformation = MeshNormalAlignmentDeformation(origin_mesh)
    mesh_deformation.update_mesh()