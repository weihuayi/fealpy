from abc import ABC, abstractmethod
from fealpy.typing import TensorLike

class ObjectiveBase(ABC):
    """优化目标函数基类"""
    
    @abstractmethod
    def fun(self, x: TensorLike) -> float:
        """计算目标函数值"""
        pass
    
    @abstractmethod
    def jac(self, x: TensorLike) -> TensorLike:
        """计算目标函数梯度"""
        pass
    
    @abstractmethod
    def hess(self, x: TensorLike, lambda_: dict) -> TensorLike:
        """计算目标函数Hessian矩阵"""
        pass