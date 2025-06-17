from fealpy.backend import backend_manager as bm
from abc import ABC, abstractmethod

class InterativeMethod(ABC):
    """Base class for iterative Navier-Stokes solvers."""
    def __init__(self):
        self.tol = 1e-8
        self.max_iter = 100
        self.validate_dependencies()

    def validate_dependencies(self):
        """Verify necessary properties and spaces"""
        required_attrs = ['_uspace', '_pspace']
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(f"FEM solver缺失必要属性: {attr}")
    
    @abstractmethod
    def BForm(self):
        pass
    
    @abstractmethod
    def LForm(self):
        pass



