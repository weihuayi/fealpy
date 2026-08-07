# 文件位置: tests/mesh/conftest.py

import pytest
import numpy as np
# 假设这是 FEALPy 中构造三角形网格的基础类，请根据实际 API 调整
# from fealpy.mesh import TriangleMesh 

@pytest.fixture(scope="function")
def standard_2d_triangle_mesh():
    """
    [全局共享夹具]
    生成一个最基础的 2D 单位正方形区域的三角形网格。
    包含 4 个节点和 2 个三角形单元。
    
    返回:
        TriangleMesh 对象
    """
    # 1. 在内存中硬编码节点坐标 (避免读取外部文件)
    # 形状为 (4, 2)，4 个顶点，二维坐标
    node = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float64)
    
    # 2. 硬编码单元拓扑结构 (逆时针顺序)
    # 形状为 (2, 3)，2 个三角形，每个三角形 3 个顶点索引
    cell = np.array([
        [1, 2, 0],
        [3, 0, 2]
    ], dtype=np.int_)
    
    # 3. 构造并返回网格对象
    # mesh = TriangleMesh(node, cell)
    # return mesh
    
    # 注意：为了基线文件不报错，这里返回一个包含 node 和 cell 的字典模拟网格对象
    # 实际项目中请替换为真实的 TriangleMesh 实例化代码
    return {"node": node, "cell": cell, "type": "Triangle"}

@pytest.fixture(scope="function")
def local_random_generator():
    """
    [全局安全辅助夹具]
    提供一个带有固定种子的局部伪随机数生成器。
    用于替代绝对禁止的全局 np.random.seed()。
    """
    return np.random.default_rng(seed=42)