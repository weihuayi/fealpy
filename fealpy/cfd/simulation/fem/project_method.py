from abc import ABC, abstractmethod
from fealpy.backend import backend_manager as bm

class ProjectionMethod(ABC):
    """投影算法抽象基类"""
    def __init__(self):
        """
        :param fem_solver: 关联的FEM主求解器实例
        """
        self.validate_dependencies()

    def validate_dependencies(self):
        """验证必要的属性和空间"""
        required_attrs = ['_uspace', '_pspace']
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(f"FEM solver缺失必要属性: {attr}")
    @abstractmethod
    def predict_velocity(self):
        """速度预测步"""
        pass
    
    @abstractmethod
    def pressure(self):
        pass
    
    @abstractmethod
    def correct_velocity(self):
        """速度校正步"""
        pass

