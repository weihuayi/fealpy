from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from fealpy.typing import TensorLike

class ObjectiveBase(ABC):
    """优化目标函数基类
    
    定义目标函数的基本接口，包括：
    1. 函数值计算
    2. 梯度计算
    3. Hessian矩阵计算
    """
    
    @abstractmethod
    def fun(self, 
            rho: TensorLike, 
            u: Optional[TensorLike] = None) -> float:
        """计算目标函数值
        
        Parameters
        ----------
        rho : 密度场
        u : 可选的位移场，如果为None则由实现类决定如何获取
        
        Returns
        -------
        value : 目标函数值
        """
        pass
    
    @abstractmethod
    def jac(self, 
            rho: TensorLike, 
            u: Optional[TensorLike] = None,
            filter_params: Optional[Dict[str, Any]] = None) -> TensorLike:
        """计算目标函数梯度
        
        Parameters
        ----------
        rho : 密度场
        u : 可选的位移场
        filter_params : 滤波器参数
        
        Returns
        -------
        gradient : 目标函数关于密度的梯度
        """
        pass
    
    @abstractmethod
    def hess(self, 
             rho: TensorLike, 
             lambda_: Dict[str, Any]) -> TensorLike:
        """计算目标函数Hessian矩阵
        
        Parameters
        ----------
        rho : 密度场
        lambda_ : Lagrange乘子相关参数
        
        Returns
        -------
        hessian : 目标函数的Hessian矩阵
        """
        pass

class ConstraintBase(ABC):
    """约束基类
    
    定义优化问题约束的基本接口，包括：
    1. 约束函数值计算
    2. 约束梯度计算
    3. 约束Hessian矩阵计算
    """
    
    @abstractmethod
    def fun(self, 
            rho: TensorLike, 
            u: Optional[TensorLike] = None) -> float:
        """计算约束函数值
        
        Parameters
        ----------
        rho : 密度场
        u : 可选的位移场
        
        Returns
        -------
        value : 约束函数值
        """
        pass
        
    @abstractmethod
    def jac(self,
            rho: TensorLike,
            u: Optional[TensorLike] = None,
            filter_params: Optional[Dict[str, Any]] = None) -> TensorLike:
        """计算约束函数梯度
        
        Parameters
        ----------
        rho : 密度场
        u : 可选的位移场
        filter_params : 滤波器参数
        
        Returns
        -------
        gradient : 约束函数关于密度的梯度
        """
        pass
        
    def hess(self, 
             rho: TensorLike, 
             lambda_: Dict[str, Any]) -> TensorLike:
        """计算约束函数Hessian矩阵
        
        Parameters
        ----------
        rho : 密度场
        lambda_ : Lagrange乘子相关参数
        
        Returns
        -------
        hessian : 约束函数的Hessian矩阵
        """
        pass

class OptimizerBase(ABC):
    """优化器基类
    
    定义优化算法的基本接口
    """
    
    @abstractmethod
    def optimize(self,
                rho: TensorLike,
                **kwargs) -> TensorLike:
        """执行优化过程
        
        Parameters
        ----------
        rho : 初始密度场
        **kwargs : 其他参数，由具体优化器定义
        
        Returns
        -------
        rho : 优化后的密度场
        """
        pass