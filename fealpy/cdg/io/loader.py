import os
from pathlib import Path
from typing import Optional, Any

from ...mesh import MFileParser, TriangleMesh


def load_mesh(file_path: str, mesh_type: Any = TriangleMesh) -> Any:
    """
    通用网格加载函数。自动根据文件后缀名选择对应的解析器。

    Args:
        file_path (str): 网格文件的路径。
        mesh_type (class, optional): 目标网格类 (例如 TriangleMesh, TetrahedralMesh)。
                                     默认为 fealpy.mesh.TriangleMesh。

    Returns:
        mesh: 实例化后的网格对象。

    Raises:
        FileNotFoundError: 如果文件不存在。
        NotImplementedError: 如果文件格式暂不支持。
        ValueError: 如果解析过程出错。
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"网格文件未找到: {file_path}")

    # 获取小写的文件后缀 (如 .m, .obj)
    ext = path.suffix.lower()

    # --- 格式分发逻辑 ---
    if ext == '.m':
        return _load_m_file(path, mesh_type)

    elif ext in ['.obj']:
        # 占位：未来实现 OBJ 读取
        raise NotImplementedError(f"目前尚未实现 {ext} 格式的读取，请等待更新。")

    elif ext in ['.off']:
        # 占位：未来实现 OFF 读取
        raise NotImplementedError(f"目前尚未实现 {ext} 格式的读取，请等待更新。")

    else:
        raise NotImplementedError(f"不支持的文件格式: {ext}")


def _load_m_file(path: Path, mesh_type: Any) -> Any:
    """内部辅助函数：调用 MFileParser 读取 .m 文件"""
    try:
        # 1. 调用我们之前写的解析器
        parser = MFileParser()
        parser.parse(str(path))

        # 2. 转换为指定的网格对象
        mesh = parser.to_mesh(mesh_type)

        return mesh

    except Exception as e:
        raise ValueError(f"解析 .m 文件时发生错误: {e}")