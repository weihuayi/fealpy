from abc import ABC, abstractmethod
from typing import Any, Optional

from fealpy.mesh import TriangleMesh
from ..operators import MeshOperator


class BaseSolver(ABC):
    """
    求解器基类 (Abstract Base Class).

    职责：
    1. 统一管理 Mesh 和 Operator 的实例。
    2. 定义所有求解器必须实现的接口。
    """

    def __init__(self, mesh: TriangleMesh):
        """
        初始化求解器。

        Args:
            mesh (TriangleMesh): 待处理的三角网格。
        """
        self.mesh = mesh
        # 自动初始化算子工厂，所有子类都可以直接用 self.operator.laplacian_matrix()
        self.operator = MeshOperator(mesh)

    @abstractmethod
    def solve(self, *args, **kwargs) -> Any:
        """
        核心求解方法。所有子类必须覆盖此方法。
        """
        pass