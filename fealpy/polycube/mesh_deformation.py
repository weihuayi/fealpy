import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from fealpy.mesh import TetrahedronMesh
from fealpy.backend import backend_manager as bm

bm.set_backend('pytorch')


def closed_axis_projection(v):
    sign = bm.sign(v).reshape(-1, 3)
    abs_max_axis = bm.argmax(bm.abs(v), axis=-1).reshape(-1)
    proj = bm.zeros_like(v).reshape(-1, 3)
    row_idx = bm.arange(proj.shape[0])
    proj[row_idx, abs_max_axis] = sign[row_idx, abs_max_axis]
    return proj.reshape(v.shape)

def compute_AMIPS_energy(node_new, node_old, cell, s=1, alpha=0.5):
    def deformation_matrix(tet_node, tet_node_new):
        p_def = tet_node_new[:, 0, :][..., None]
        def_edges = bm.concat([
            p_def - tet_node_new[:, 1, :][..., None],
            p_def - tet_node_new[:, 2, :][..., None],
            p_def - tet_node_new[:, 3, :][..., None]
        ], axis=-1)
        p_orig = tet_node[:, 0, :][..., None]
        orig_edges = bm.concat([
            p_orig - tet_node[:, 1, :][..., None],
            p_orig - tet_node[:, 2, :][..., None],
            p_orig - tet_node[:, 3, :][..., None]
        ], axis=-1)
        inv_orig_edges = bm.linalg.inv(orig_edges)
        A = bm.einsum('nij,njk->nik', def_edges, inv_orig_edges)
        return A

    tet_node = node_old[cell]
    tet_node_new = node_new[cell]
    A = deformation_matrix(tet_node, tet_node_new)
    A_inv = bm.linalg.inv(A)
    delta_conf = bm.einsum('nij->n', A**2) * bm.einsum('nij->n', A_inv**2) / 8
    delta_vol = bm.linalg.det(A) * bm.linalg.det(A_inv) / 2
    e_iso = bm.exp(s * (alpha * delta_conf + (1 - alpha) * delta_vol))
    return e_iso

def phi(x):
    nx = x[..., 0]**2
    ny = x[..., 1]**2
    nz = x[..., 2]**2
    return nx * ny + ny * nz + nz * nx


class MeshOptimizationBase:
    """
    The base class for mesh optimization algorithms.
    This class provides a framework for adding optimizable parameters,
    performing optimization steps, and managing the optimization process.

    Parameters:
        lr: float
            Learning rate for the optimizers.
        max_epochs: int
            Maximum number of optimization epochs.
        error_threshold: float
            Threshold for stopping the optimization based on parameter changes.
        weights: dict, optional
            A dictionary of weights for different optimizable parameters.

    Attributes:
        optimizers: dict
            A dictionary mapping parameter names to their optimizers.
        parameters: dict
            A dictionary mapping parameter names to their corresponding tensors.
        prev_parameters: dict
            A dictionary to store the previous values of the parameters for convergence checks.
    """
    def __init__(self, lr=0.001, max_epochs=10000, error_threshold=1e-3, weights=None):
        self.lr = lr
        self.max_epochs = max_epochs
        self.error_threshold = error_threshold
        self.optimizers = {}
        self.parameters = {}
        self.prev_parameters = {}
        # 初始化权重字典，默认为1.0
        self.weights = weights if weights is not None else {}

    def add_optimizable(self, name, param, optimizer_type=torch.optim.Adam):
        self.parameters[name] = param
        self.optimizers[name] = optimizer_type([param], lr=self.lr)
        self.prev_parameters[name] = param.detach().clone()
        # 如果权重未定义，为新参数设置默认权重1.0
        if name not in self.weights:
            self.weights[name] = 1.0

    def optimization_step(self):
        raise NotImplementedError("Subclasses must implement optimization_step")

    def optimize(self):
        pre_energy = float('inf')
        for step in range(self.max_epochs):
            energy = self.optimization_step()
            if (step + 1) % 50 == 0:
                print(f"Step [{step + 1}], Energy: {energy.item():.4f}")

            param_diff = 0.0
            for name, param in self.parameters.items():
                diff = bm.linalg.norm(param - self.prev_parameters[name])
                param_diff += self.weights[name] * diff

            for name, param in self.parameters.items():
                self.prev_parameters[name] = param.detach().clone()

            if (step == self.max_epochs - 1) or (param_diff < self.error_threshold and energy > pre_energy):
                break
            pre_energy = energy

        return self.get_optimized_result()

    def get_optimized_result(self):
        raise NotImplementedError("Subclasses must implement get_optimized_result")

    def plot_mesh(self, mesh, title="Mesh"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        mesh.add_plot(ax)
        ax.set_title(title)
        plt.show()

class MeshNormalSmoothDeformation(MeshOptimizationBase):
    """
    A class for mesh deformation based on normal smoothing.

    Parameters:
        origin_mesh: TetrahedronMesh
            The original mesh to be deformed.
        sigma: float
            Standard deviation for Gaussian smoothing.
        s: float
            Scaling factor for the AMIPS energy.
        alpha: float
            Weighting factor for the AMIPS energy.
        lr: float
            Learning rate for the optimization.
        max_epochs: int
            Maximum number of optimization epochs.
        error_threshold: float
            Threshold for stopping the optimization based on parameter changes.
        weights: dict, optional
            A dictionary of weights for different optimizable parameters.
    """
    def __init__(self, origin_mesh, sigma=0.1, s=1, alpha=0.5, lr=0.001, max_epochs=10000, error_threshold=1e-3, weights=None):
        super().__init__(lr, max_epochs, error_threshold, weights)
        self.origin_mesh = origin_mesh
        self.sigma, self.s, self.alpha = sigma, s, alpha
        self.cell = origin_mesh.cell
        self.new_node = bm.zeros_like(origin_mesh.node, dtype=bm.float64, requires_grad=True)
        self.new_node.data = origin_mesh.node.data
        self.old_node = origin_mesh.node
        self.rotate_matrix = bm.eye(3, dtype=bm.float64, requires_grad=True)
        self.add_optimizable("node", self.new_node)
        self.add_optimizable("rotate_matrix", self.rotate_matrix)

    def compute_gaussian_normals(self, face_centers, areas, face_normal):
        t1 = -bm.linalg.norm(face_centers[:, None, :] - face_centers[None, ...], axis=-1) ** 2 / (2 * self.sigma ** 2)
        t2 = bm.exp(t1)
        gaussian_normals = bm.einsum('j,ij,jd->id', areas, t2, face_normal)
        gaussian_normals = gaussian_normals / bm.linalg.norm(gaussian_normals, axis=-1, keepdims=True)
        return gaussian_normals

    def compute_smooth_normal_energy(self, gauss_normals, origin_normals):
        diff = origin_normals - closed_axis_projection(gauss_normals)
        energy = bm.linalg.norm(diff, axis=-1) ** 2
        return energy

    def compute_e_o(self, face_normal):
        rotate_normal = bm.einsum('ij,...j->...i', self.rotate_matrix, face_normal)
        return bm.sum(phi(rotate_normal))

    def compute_e_ns(self, face_centers, areas, face_normal, face2cell):
        face_normal = bm.einsum('ij,...j->...i', self.rotate_matrix, face_normal)
        gaussian_normals = self.compute_gaussian_normals(face_centers, areas, face_normal)
        e_s_elem = self.compute_smooth_normal_energy(gaussian_normals, face_normal)
        e_iso_elem = compute_AMIPS_energy(self.new_node, self.old_node, self.cell, self.s, self.alpha)
        mu_min, mu_max = bm.array(1e3), bm.array(1e16)
        mu = bm.minimum(mu_max, bm.maximum(mu_min, e_iso_elem[face2cell] / e_s_elem))
        e_s = bm.einsum('i,i->', mu, e_s_elem)
        e_iso = bm.sum(e_iso_elem)
        return e_s + e_iso

    def optimization_step(self):
        mesh = TetrahedronMesh(self.new_node, self.cell)
        bd_face_idx = mesh.boundary_face_index()
        face_centers = mesh.entity_barycenter('face', bd_face_idx)
        areas = mesh.entity_measure('face', bd_face_idx)
        face2cell = mesh.face2cell[bd_face_idx, 0]
        face_normal = mesh.face_unit_normal(bd_face_idx)

        # 优化 rotate_matrix
        self.optimizers["rotate_matrix"].zero_grad()
        energy_o = self.compute_e_o(face_normal)
        energy_o.backward(retain_graph=True)
        self.optimizers["rotate_matrix"].step()

        # 优化 node
        self.optimizers["node"].zero_grad()
        energy_ns = self.compute_e_ns(face_centers, areas, face_normal, face2cell)
        energy_ns.backward()
        self.old_node = self.new_node.detach().clone()
        self.optimizers["node"].step()

        # return energy_o + energy_ns
        return energy_ns

    def get_optimized_result(self):
        optimized_mesh = TetrahedronMesh(self.new_node, self.cell)
        # face_normal = optimized_mesh.face_unit_normal(optimized_mesh.boundary_face_index())
        # t = bm.einsum('ij,nj->ni', self.rotate_matrix, face_normal)
        # print(face_normal)
        # print(t / bm.linalg.norm(t, axis=1).reshape(-1, 1))
        return optimized_mesh

class MeshNormalAlignmentDeformation(MeshOptimizationBase):
    """
    A class for mesh deformation based on normal alignment.

    Parameters:
        origin_mesh: TetrahedronMesh
            The original mesh to be deformed.
        gamma: float
            Scaling factor for the alignment energy.
        s: float
            Scaling factor for the AMIPS energy.
        alpha: float
            Weighting factor for the AMIPS energy.
        lr: float
            Learning rate for the optimization.
        max_epochs: int
            Maximum number of optimization epochs.
        error_threshold: float
            Threshold for stopping the optimization based on parameter changes.
        weights: dict, optional
            A dictionary of weights for different optimizable parameters.
    """
    axes = bm.tensor([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=bm.float64)

    def __init__(self, origin_mesh, gamma=1e3, s=1, alpha=0.5, lr=0.001, max_epochs=10000, error_threshold=1e-5, weights=None):
        super().__init__(lr, max_epochs, error_threshold, weights)
        self.origin_mesh = origin_mesh
        self.gamma, self.s, self.alpha = gamma, s, alpha
        self.cell = origin_mesh.cell
        self.new_node = bm.zeros_like(origin_mesh.node, dtype=bm.float64, requires_grad=True)
        self.new_node.data = origin_mesh.node.data
        self.old_node = origin_mesh.node
        self.rotate_matrix = bm.eye(3, dtype=bm.float64, requires_grad=True)
        self.add_optimizable("node", self.new_node)
        self.add_optimizable("rotate_matrix", self.rotate_matrix)

    def compute_e_o(self, face_normal):
        rotate_normal = bm.einsum('ij,...j->...i', self.rotate_matrix, face_normal)
        return bm.sum(phi(rotate_normal))

    def compute_e_na(self, face_normal, face2cell):
        r_n = (self.rotate_matrix @ face_normal.T).T
        phi_n = phi(r_n)
        e_iso_elem = compute_AMIPS_energy(self.new_node, self.old_node, self.cell, self.s, self.alpha)
        omega_min, omega_max = bm.array(1e3), bm.array(1e16)
        omega = bm.minimum(omega_max, self.gamma * bm.maximum(omega_min, e_iso_elem[face2cell] / phi_n))
        e_a = bm.sum(omega.reshape(-1) * phi_n)
        e_iso = bm.sum(e_iso_elem)
        return e_iso + e_a

    def optimization_step(self):
        mesh = TetrahedronMesh(self.new_node, self.cell)
        bd_face_idx = mesh.boundary_face_index()
        face_normal = mesh.face_unit_normal(bd_face_idx)
        face2cell = mesh.face2cell[bd_face_idx, 0]

        # 优化 rotate_matrix
        self.optimizers["rotate_matrix"].zero_grad()
        energy_o = self.compute_e_o(face_normal)
        energy_o.backward(retain_graph=True)
        self.optimizers["rotate_matrix"].step()

        # 优化 node
        self.optimizers["node"].zero_grad()
        energy_na = self.compute_e_na(face_normal, face2cell)
        energy_na.backward()
        self.old_node = self.new_node.detach().clone()
        self.optimizers["node"].step()

        # return energy_o + energy_na
        return energy_na

    def get_optimized_result(self):
        optimized_mesh = TetrahedronMesh(self.new_node, self.cell)
        # face_normal = optimized_mesh.face_unit_normal(optimized_mesh.boundary_face_index())
        # t = bm.einsum('ij,nj->ni', self.rotate_matrix, face_normal)
        # print(face_normal)
        # print(t / bm.linalg.norm(t, axis=1).reshape(-1, 1))
        return optimized_mesh

class MeshBallDeformation(MeshOptimizationBase):
    def __init__(self, origin_mesh: TetrahedronMesh, s=1, alpha=0.5, lr=0.001,
                 max_epochs=10000, error_threshold=1e-3,
                 weights=None, weight_amips=1.0, weight_ball=1.0):
        super().__init__(lr, max_epochs, error_threshold, weights)
        self.origin_mesh = origin_mesh
        self.s, self.alpha = s, alpha
        self.cell = origin_mesh.cell
        self.new_node = bm.zeros_like(origin_mesh.node, dtype=bm.float64, requires_grad=True)
        self.new_node.data = origin_mesh.node.data
        self.old_node = origin_mesh.node
        self.add_optimizable("node", self.new_node)
        self.weight_amips = weight_amips
        self.weight_ball = weight_ball

    def compute_center_and_radius(self):
        """计算网格重心和球面半径"""
        center = bm.mean(self.new_node, axis=0)  # 重心
        max_distance = bm.max(bm.linalg.norm(self.new_node - center, axis=1))  # 最大距离
        radius = 2 * max_distance  # 半径为最大距离的两倍
        return center, radius

    def compute_ball_energy(self, node, center, radius):
        """计算节点到球面的距离能量"""
        distances = bm.linalg.norm(node - center, axis=1)
        # 为边界节点添加更高权重（假设 bd_node_idx 已计算）
        bd_node_idx = self.origin_mesh.boundary_node_index()
        weights = bm.ones(node.shape[0])
        weights[bd_node_idx] = 2.0  # 边界节点权重加倍
        energy = bm.sum(weights * (distances - radius) ** 2)
        return energy

    def compute_energy(self):
        """计算总能量：AMIPS 能量 + 球面距离能量"""
        mesh = TetrahedronMesh(self.new_node, self.cell)
        e_iso = compute_AMIPS_energy(self.new_node, self.old_node, self.cell, self.s, self.alpha)
        center, radius = self.compute_center_and_radius()
        e_ball = self.compute_ball_energy(self.new_node, center, radius)
        return self.weight_amips * bm.sum(e_iso) + self.weight_ball * e_ball

    def optimization_step(self):
        """优化步进"""
        self.optimizers["node"].zero_grad()
        energy = self.compute_energy()
        energy.backward()
        self.old_node = self.new_node.detach().clone()
        self.optimizers["node"].step()
        return energy

    def get_optimized_result(self):
        """返回优化后的网格"""
        optimized_mesh = TetrahedronMesh(self.new_node, self.cell)
        return optimized_mesh

if __name__ == "__main__":
    origin_mesh = pickle.load(open("../../example/polycube/data/unit_sphere_mesh_torch.pkl", "rb"))
    weights = {"node": 1.0, "rotate_matrix": 0.0}
    deformer = MeshNormalSmoothDeformation(origin_mesh,
                                           sigma=0.1, s=1, alpha=0.5, max_epochs=100000,
                                           error_threshold=1e-3, weights=weights)
    # deformer = MeshNormalAlignmentDeformation(origin_mesh,
    #                                           gamma=1e3, s=1, alpha=0.5, max_epochs=100000,
    #                                           error_threshold=1e-3, weights=weights)
    optimized_mesh = deformer.optimize()
    deformer.plot_mesh(optimized_mesh, title="Optimized Mesh")