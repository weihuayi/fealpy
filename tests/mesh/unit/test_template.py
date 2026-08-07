# 文件位置: tests/mesh/unit/test_template.py

import pytest
import numpy as np
# from fealpy.mesh import TriangleMesh

class TestMeshTemplate:
    """
    FEALPy 网格模块单元测试标准模板
    同学们可以直接复制此类的结构来编写新的测试。
    """

    def test_euler_formula_topology(self, standard_2d_triangle_mesh):
        """
        [规范演示]：数学不变量断言
        测试拓扑守恒定律：二维简单连通网格必须满足欧拉公式 V - E + F = 1
        """
        mesh = standard_2d_triangle_mesh
        
        # 假设这里调用了 FEALPy 的内置方法获取节点、边、面的数量
        # V = mesh.number_of_nodes()
        # E = mesh.number_of_edges()
        # F = mesh.number_of_cells()
        
        # 模拟数据验证
        V = 4
        E = 5
        F = 2
        
        # 拒绝魔法数字 (如 assert 1 == 1)，坚持数学不变量：顶点数 - 边数 + 面数 = 欧拉示性数
        euler_characteristic = V - E + F
        assert euler_characteristic == 1, "网格拓扑结构损坏，不满足欧拉公式！"

    @pytest.mark.parametrize(
        "precision_type, expected_dtype",
        [
            ("float32", np.float32),
            ("float64", np.float64)
        ],
        ids=["precision-float32", "precision-float64"] # <-- [高压红线]：必须指定 ids！
    )
    def test_node_precision(self, standard_2d_triangle_mesh, precision_type, expected_dtype):
        """
        [规范演示]：参数化测试与 ids 命名
        测试网格节点坐标的数据精度转换功能。
        如果这里报错，CI 会精准显示：test_node_precision[precision-float64]
        """
        mesh = standard_2d_triangle_mesh
        node = mesh["node"]
        
        # 模拟精度转换逻辑
        node_converted = node.astype(precision_type)
        
        # 断言数据类型
        assert node_converted.dtype == expected_dtype
        # 比对浮点数矩阵，严禁使用 ==，必须使用 assert_allclose 以吸收精度误差
        np.testing.assert_allclose(node_converted, node, atol=1e-7)

    def test_mesh_export_safe(self, standard_2d_triangle_mesh, tmp_path):
        """
        [规范演示]：安全的文件读写
        测试网格导出功能，绝对禁止在当前目录留垃圾。
        """
        mesh = standard_2d_triangle_mesh
        
        # tmp_path 是 pytest 提供的沙箱目录，用完即焚
        export_dir = tmp_path / "export_test"
        export_dir.mkdir()
        file_path = export_dir / "test_mesh.vtk"
        
        # 模拟导出操作
        # mesh.to_vtk(file_path)
        file_path.write_text("mock vtk content") # 模拟写入
        
        # 断言文件确实生成了，且大小大于 0
        assert file_path.exists()
        assert file_path.stat().st_size > 0

    def test_random_noise_addition(self, standard_2d_triangle_mesh, local_random_generator):
        """
        [规范演示]：局部随机数隔离
        测试给节点添加噪音，绝不允许使用 np.random.seed()。
        """
        mesh = standard_2d_triangle_mesh
        node = mesh["node"]
        
        # 使用夹具提供的局部生成器，保证结果 100% 可复现，且不污染全局
        noise = local_random_generator.normal(loc=0.0, scale=0.1, size=node.shape)
        noisy_node = node + noise
        
        # 断言噪音确实被加上了 (两矩阵不相等)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(node, noisy_node)