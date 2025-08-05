from abc import ABC, abstractmethod

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.fem import DirichletBC
from fealpy.fem import LinearForm, SourceIntegrator, BlockForm
from fealpy.sparse import COOTensor

from .fem_base import FEM

class ProjectionMethod(FEM, ABC):
    """投影算法抽象基类"""
    def __init__(self, equation, mesh):
        super().__init__(equation, mesh)
        
        if self.equation.pressure_neumann == False:
            self.threshold = self.equation.pde.is_pressure_boundary


    @abstractmethod
    def predict_velocity(self):
        """速度预测步"""
        pass
    
    @abstractmethod
    def pressure(self):
        """压力求解步"""
        pass
    
    @abstractmethod
    def correct_velocity(self):
        """速度校正步"""
        pass

    @variantmethod("dirichlet")
    def apply_bc(self, A, b, pde):
        BC = DirichletBC(
            (self.uspace, self.pspace), 
            gd=(lambda p: self.velocity(p, self.timeline.next_time()), lambda p: self.pde.pressure(p, self.timeline.next_time())), 
            threshold=(self.pde.is_velocity_boundary, self.pde.is_pressure_boundary),
            method='interp')
        A, b = BC.apply(A, b)
        return A, b
    
    def lagrange_multiplier(self, A, b, c=0):
        """
        Constructs the augmented system matrix for Lagrange multipliers.
        c is the integral of pressure, default is 0.
        """

        LagLinearForm = LinearForm(self.pspace)
        LagLinearForm.add_integrator(SourceIntegrator(source=1))
        LagA = LagLinearForm.assembly()

        A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
                                 bm.arange(len(LagA), dtype=bm.int32)]), LagA, spshape=(1, len(LagA)))

        A = BlockForm([[A, A1.T], [A1, None]])
        A = A.assembly_sparse_matrix(format='csr')
        b0 = bm.array([c])
        b  = bm.concatenate([b, b0], axis=0)
        return A, b
