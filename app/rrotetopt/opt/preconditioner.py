from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.operator import LinearOperator
from fealpy.solver import cg

class BasePreconditioner:
    """
    通用预条件子接口。
    优化器只依赖这个接口，不依赖具体矩阵形式。
    """

    def __init__(self, update_interval: int = 1) -> None:
        self.update_interval = update_interval

    def setup(self, x: TensorLike) -> None:
        """
        首次初始化内部结构。
        """
        pass

    def update(self, x: TensorLike, k: Optional[int] = None) -> None:
        """
        按需更新内部结构。
        """
        pass

    def apply(self, q: TensorLike) -> TensorLike:
        """
        返回 M^{-1} q
        """
        return q

    def scale(self, s: TensorLike, y: TensorLike) -> float:
        """
        给 L-BFGS 初始Hessian缩放用。
        默认回退到标准 gamma = (s^T y)/(y^T y)
        """
        yty = float(y @ y)
        if yty > 0:
            return float((s @ y) / yty)
        return 1.0

    def reset(self) -> None:
        """
        拓扑变化后重置内部状态。
        """
        pass


