from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh, TetrahedronMesh
import torch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


bm.set_backend('pytorch')
torch.set_printoptions(threshold=float('inf'))

import torch


class TetrahedralMeshDeformer:
    """一个用于随机变形四面体网格的类，确保无负体积单元且单元质量较高。"""

    def __init__(self, origin_mesh: TetrahedronMesh, scale_factor=0.1, min_quality=0.01, max_attempts=10):
        """
        初始化四面体网格变形器。

        参数：
            origin_mesh: TetrahedronMesh 对象，包含节点和单元信息
            scale_factor: 扰动幅度系数（相对于局部网格尺寸）
            min_quality: 单元质量阈值
            max_attempts: 最大尝试次数
        """
        self.nodes = origin_mesh.node.clone().to(dtype=torch.float64)
        self.cell = origin_mesh.cell.clone().to(dtype=torch.int32)
        self.scale_factor = scale_factor
        self.min_quality = min_quality
        self.max_attempts = max_attempts
        self.device = self.nodes.device

        # 验证输入
        if self.nodes.shape[1] != 3:
            raise ValueError("Nodes must have shape (N, 3)")
        if self.cell.shape[1] != 4:
            raise ValueError("Cell must have shape (M, 4)")

        # 初始化边界节点（可选）
        self.boundary_nodes = None

    def tetrahedron_volume(self, nodes, cell):
        """
        计算四面体单元的体积。

        参数：
            nodes: (N, 3) 张量，网格点坐标
            cell: (M, 4) 张量，四面体单元的节点索引

        返回：
            volumes: (M,) 张量，每个单元的体积
        """
        a = nodes[cell[:, 0]]  # (M, 3)
        b = nodes[cell[:, 1]]  # (M, 3)
        c = nodes[cell[:, 2]]  # (M, 3)
        d = nodes[cell[:, 3]]  # (M, 3)

        ad = a - d
        bd = b - d
        cd = c - d

        cross = torch.cross(bd, cd, dim=1)
        dot = torch.sum(ad * cross, dim=1)
        volume = torch.abs(dot) / 6.0
        return volume

    def tetrahedron_quality(self, nodes, cell):
        """
        计算四面体单元的质量（基于体积与最大边长）。

        参数：
            nodes: (N, 3) 张量，网格点坐标
            cell: (M, 4) 张量，四面体单元的节点索引

        返回：
            qualities: (M,) 张量，每个单元的质量（越大越好）
        """
        a = nodes[cell[:, 0]]
        b = nodes[cell[:, 1]]
        c = nodes[cell[:, 2]]
        d = nodes[cell[:, 3]]

        edges = [
            torch.norm(a - b, dim=1),
            torch.norm(a - c, dim=1),
            torch.norm(a - d, dim=1),
            torch.norm(b - c, dim=1),
            torch.norm(b - d, dim=1),
            torch.norm(c - d, dim=1)
        ]
        max_edge = torch.stack(edges, dim=1).max(dim=1)[0]

        volume = self.tetrahedron_volume(nodes, cell)
        quality = volume / (max_edge ** 3 + 1e-10)
        return quality

    def compute_local_scale(self):
        """
        计算每个节点的局部网格尺寸（到邻居的平均距离）。

        返回：
            scales: (N,) 张量，每个节点的局部尺寸
        """
        N = self.nodes.shape[0]
        scales = torch.zeros(N, dtype=self.nodes.dtype, device=self.device)
        counts = torch.zeros(N, dtype=torch.int32, device=self.device)

        for i in range(4):
            for j in range(i + 1, 4):
                ni = self.cell[:, i]
                nj = self.cell[:, j]
                dist = torch.norm(self.nodes[ni] - self.nodes[nj], dim=1)
                scales.index_add_(0, ni, dist)
                scales.index_add_(0, nj, dist)
                counts.index_add_(0, ni, torch.ones_like(ni))
                counts.index_add_(0, nj, torch.ones_like(nj))

        scales = scales / (counts.float() + 1e-10)
        return scales

    def detect_boundary_nodes(self):
        """
        检测网格的边界节点（出现在奇数次面的节点）。

        返回：
            boundary_nodes: 张量，边界节点的索引
        """
        faces = torch.cat([
            self.cell[:, [0, 1, 2]],
            self.cell[:, [0, 1, 3]],
            self.cell[:, [0, 2, 3]],
            self.cell[:, [1, 2, 3]]
        ])

        face_counts = torch.zeros(self.nodes.shape[0], dtype=torch.int32, device=self.device)
        for face in faces:
            for node in face:
                face_counts[node] += 1

        boundary_nodes = (face_counts % 2 == 1).nonzero(as_tuple=True)[0]
        return boundary_nodes

    def perturb_nodes(self, nodes, fix_boundary=True):
        """
        随机扰动节点位置，幅度基于局部网格尺寸。

        参数：
            nodes: (N, 3) 张量，网格点坐标
            fix_boundary: 是否固定边界节点

        返回：
            new_nodes: (N, 3) 张量，扰动后的节点坐标
        """
        scales = self.compute_local_scale()
        perturbation = torch.rand_like(nodes) * 2.0 - 1.0
        perturbation = perturbation * scales[:, None] * self.scale_factor

        if fix_boundary:
            if self.boundary_nodes is None:
                self.boundary_nodes = self.detect_boundary_nodes()
            perturbation[self.boundary_nodes] = 0

        new_nodes = nodes + perturbation
        return new_nodes

    def laplacian_smoothing(self, nodes, iterations=1, fix_boundary=True):
        """
        应用拉普拉斯平滑优化网格。

        参数：
            nodes: (N, 3) 张量，网格点坐标
            iterations: 平滑迭代次数
            fix_boundary: 是否固定边界节点

        返回：
            smoothed_nodes: (N, 3) 张量，平滑后的节点坐标
        """
        if fix_boundary and self.boundary_nodes is None:
            self.boundary_nodes = self.detect_boundary_nodes()

        for _ in range(iterations):
            new_nodes = torch.zeros_like(nodes)
            counts = torch.zeros(nodes.shape[0], dtype=torch.int32, device=self.device)

            for i in range(4):
                for j in range(i + 1, 4):
                    ni = self.cell[:, i]
                    nj = self.cell[:, j]
                    new_nodes.index_add_(0, ni, nodes[nj])
                    new_nodes.index_add_(0, nj, nodes[ni])
                    counts.index_add_(0, ni, torch.ones_like(ni))
                    counts.index_add_(0, nj, torch.ones_like(nj))

            new_nodes = new_nodes / (counts[:, None].float() + 1e-10)

            if fix_boundary:
                new_nodes[self.boundary_nodes] = nodes[self.boundary_nodes]

            nodes = new_nodes

        return nodes

    def deform(self, smoothing_iterations=1, fix_boundary=True):
        """
        执行随机网格变形，避免负体积和低质量单元。

        参数：
            smoothing_iterations: 拉普拉斯平滑迭代次数
            fix_boundary: 是否固定边界节点

        返回：
            new_nodes: (N, 3) 张量，变形后的节点坐标
            success: bool，是否成功生成有效网格
        """
        original_nodes = self.nodes.clone()

        for attempt in range(self.max_attempts):
            # 随机扰动
            new_nodes = self.perturb_nodes(original_nodes, fix_boundary)

            # 检查体积
            volumes = self.tetrahedron_volume(new_nodes, self.cell)
            if torch.any(volumes <= 0):
                continue

            # 检查质量
            qualities = self.tetrahedron_quality(new_nodes, self.cell)
            if torch.any(qualities < self.min_quality):
                continue

            # 应用拉普拉斯平滑
            if smoothing_iterations > 0:
                new_nodes = self.laplacian_smoothing(new_nodes, smoothing_iterations, fix_boundary)

                # 再次检查体积和质量
                volumes = self.tetrahedron_volume(new_nodes, self.cell)
                if torch.any(volumes <= 0):
                    continue

                qualities = self.tetrahedron_quality(new_nodes, self.cell)
                if torch.any(qualities < self.min_quality):
                    continue

            return TetrahedronMesh(new_nodes, self.cell), True

        print(f"Warning: Failed to generate valid mesh after {self.max_attempts} attempts.")
        return TetrahedronMesh(new_nodes, self.cell), False

    def validate_mesh(self, nodes):
        """
        验证网格是否有效（无负体积，质量达标）。

        参数：
            nodes: (N, 3) 张量，网格点坐标

        返回：
            is_valid: bool，网格是否有效
            min_volume: float，最小体积
            min_quality: float，最小质量
        """
        volumes = self.tetrahedron_volume(nodes, self.cell)
        qualities = self.tetrahedron_quality(nodes, self.cell)

        is_valid = (volumes > 0).all() and (qualities >= self.min_quality).all()
        min_volume = volumes.min().item()
        min_quality = qualities.min().item()

        return is_valid, min_volume, min_quality


# 示例使用
if __name__ == "__main__":
    origin_mesh = TetrahedronMesh.from_box(nx=3, ny=3, nz=3)

    # 创建变形器实例
    deformer = TetrahedralMeshDeformer(origin_mesh, scale_factor=0.1, min_quality=0.01, max_attempts=100)
    # 执行变形
    deformed_mesh, success = deformer.deform(smoothing_iterations=2, fix_boundary=True)
    # 验证网格
    is_valid, min_volume, min_quality = deformer.validate_mesh(deformed_mesh.node)
    print(f"Deformation {'successful' if success else 'failed'}")
    print(f"Mesh valid: {is_valid}, Min volume: {min_volume:.6f}, Min quality: {min_quality:.6f}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    origin_mesh.add_plot(ax)
    ax.set_title("Original Mesh")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    deformed_mesh.add_plot(ax)
    ax.set_title("Deformed Mesh")
    plt.show()
