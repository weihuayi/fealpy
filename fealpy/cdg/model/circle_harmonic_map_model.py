from ...backend import bm
from ...model import ComputationalModel
from ...mesh import TriangleMesh, MFileParser
from ...functionspace import LagrangeFESpace
from ...fem import (BilinearForm, DirichletBC, LinearForm,
                        ScalarDiffusionIntegrator, ScalarSourceIntegrator)
from ...solver import spsolve

# 导入同目录下的 PDE 定义
from .pde import HarmonicMapPDE


class CircleHarmonicMapModel(ComputationalModel):
    """
    计算共形几何：调和映射模型 (Harmonic Map Model)
    将 3D 曲面网格映射到 2D 平面（如单位圆）。
    """

    def __init__(self, options):
        # 初始化父类 (日志记录等)
        super().__init__(
            pbar_log=options.get('pbar_log', False),
            log_level=options.get('log_level', 'INFO')
        )

        self.options = options
        self.mesh_path = options['mesh_path']
        self.space_degree = options.get('degree', 1)
        self.integration_q = options.get('integration_q', 3)  # 积分精度

        # 1. 初始化 PDE 和 网格
        self.pde = HarmonicMapPDE()
        self.set_mesh()

        # 2. 建立空间
        self.set_space()

    def __str__(self) -> str:
        s = f"{self.__class__.__name__}(\n"
        s += "  --- Harmonic Map Model ---\n"
        s += f"  mesh file      : {self.mesh_path}\n"
        s += f"  mesh type      : {self.mesh.__class__.__name__}\n"
        s += f"  cells          : {self.mesh.number_of_cells()}\n"
        s += f"  nodes          : {self.mesh.number_of_nodes()}\n"
        s += f"  space degree   : {self.space_degree}\n"
        s += ")"
        return s

    def set_mesh(self) -> None:
        """加载数据并生成网格"""
        file_parser = MFileParser()
        parser = file_parser.parse(self.mesh_path)
        self.mesh = parser.to_mesh(TriangleMesh)
        self.logger.info(f"Mesh loaded from {self.mesh_path}")

    def set_space(self) -> None:
        """建立有限元函数空间"""
        self.space = LagrangeFESpace(self.mesh, p=self.space_degree)
        self.logger.info(f"Function space created with degree p={self.space_degree}")

    def assemble_system(self):
        """
        组装刚度矩阵 A 和 载荷向量 F。
        注意：对于调和映射，u 和 v 坐标分量共享相同的刚度矩阵 A，
        因为算子都是 Laplace 算子。
        """
        # 组装矩阵 A
        bform = BilinearForm(self.space)
        bform.add_integrator(ScalarDiffusionIntegrator(q=self.integration_q))
        A = bform.assembly()

        # 组装右端项 F (源项通常为0)
        lform = LinearForm(self.space)
        lform.add_integrator(ScalarSourceIntegrator(self.pde.source, q=self.integration_q))
        F = lform.assembly()

        return A, F

    def compute_boundary_mapping(self):
        """
        计算边界映射: 将 3D 网格的边界映射到 2D 单位圆。

        Returns:
            bd_idx: 边界节点的全局索引
            uv_boundary: 对应边界节点在 2D 圆上的坐标 (N_bd, 2)
        """
        mesh = self.mesh
        nodes = mesh.entity('node')

        # 获取有序的边界节点索引
        bd_idx = mesh.order_edge(start_num=0)

        # 提取边界节点坐标
        bd_coords = nodes[bd_idx]

        # 计算边界线段长度
        # 注意：最后一个点要连回第一个点
        next_coords = nodes[bm.roll(bd_idx, -1)]
        segment_lengths = bm.linalg.norm(next_coords - bd_coords, axis=1)

        # 计算累积长度（弧长参数化）
        total_length = bm.sum(segment_lengths)
        cumulative_length = bm.concatenate(([0], bm.cumsum(segment_lengths)))

        # 映射到单位圆
        # 这里的 cumulative_length 比 bd_idx 多一个元素(0和最后总长)，
        # 我们只需要前 N 个对应 bd_idx
        theta = 2.0 * bm.pi * cumulative_length[:-1] / total_length

        u_bd = bm.cos(theta)
        v_bd = bm.sin(theta)

        uv_boundary = bm.stack([u_bd, v_bd], axis=-1)

        return bd_idx, uv_boundary

    def solve(self):
        """求解过程"""

        # 1. 获取原始系统的 A 和 F
        A_base, F_base = self.assemble_system()

        # 2. 计算边界条件 (映射到圆)
        bd_idx, uv_boundary = self.compute_boundary_mapping()

        # 3. 准备结果容器
        uh_u = self.space.function()  # u 分量 (x坐标)
        uh_v = self.space.function()  # v 分量 (y坐标)

        gdof = self.space.number_of_global_dofs()

        # --- 求解 U 分量 ---
        dirichlet_u = bm.zeros(gdof)
        dirichlet_u[bd_idx] = uv_boundary[:, 0]

        bc_u = DirichletBC(self.space, dirichlet_u, threshold=bd_idx)
        # 注意：这里直接传 threshold=bd_idx (索引数组) 在某些版本fealpy可能需要转化为bool mask
        # 我们可以构建一个 bool mask 来兼容
        is_bd = bm.zeros(gdof, dtype=bool)
        is_bd[bd_idx] = True
        bc_u = DirichletBC(self.space, dirichlet_u, threshold=is_bd)

        A1, F1 = bc_u.apply(A_base.copy(), F_base.copy())  # 复制一份，避免修改原矩阵
        uh_u[:] = spsolve(A1, F1)

        # --- 求解 V 分量 ---
        dirichlet_v = bm.zeros(gdof)
        dirichlet_v[bd_idx] = uv_boundary[:, 1]

        bc_v = DirichletBC(self.space, dirichlet_v, threshold=is_bd)

        A2, F2 = bc_v.apply(A_base.copy(), F_base.copy())
        uh_v[:] = spsolve(A2, F2)

        self.logger.info("Solved harmonic map for both U and V components.")
        return uh_u, uh_v

    def show(self, uh_u, uh_v, save_dir='./results', prefix='girl'):
        """结果导出与可视化"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        # 1. 导出原始 3D 网格
        original_vtu = os.path.join(save_dir, f'{prefix}_original.vtu')
        self.mesh.to_vtk(fname=original_vtu)
        self.logger.info(f"Saved original mesh to {original_vtu}")

        # 2. 构造参数化后的 2D 网格
        # 使用求解出的 (u, v) 作为新的节点坐标，z 轴设为 0
        uv_coords = bm.stack([uh_u.array, uh_v.array], axis=-1)

        # 注意：FEALPy 的 TriangleMesh 通常需要和 spatial dimension 一致的坐标
        # 如果是纯 2D 网格，可以直接传 (N, 2)。

        circle_mesh = TriangleMesh(uv_coords, self.mesh.ds.cell)

        result_vtu = os.path.join(save_dir, f'{prefix}_circle.vtu')
        circle_mesh.to_vtk(fname=result_vtu)
        self.logger.info(f"Saved parameterized circle mesh to {result_vtu}")