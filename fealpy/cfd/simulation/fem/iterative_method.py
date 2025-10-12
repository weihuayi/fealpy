from abc import ABC, abstractmethod

from fealpy.backend import backend_manager as bm
from fealpy.fem import LinearForm, SourceIntegrator, BlockForm
from fealpy.sparse import COOTensor
from fealpy.decorator import variantmethod
from fealpy.fem import DirichletBC
from fealpy.decorator import cartesian

from .fem_base import FEM


class IterativeMethod(FEM, ABC):
    """
    Abstract base class for iterative solvers in the finite element method (FEM).
    This class inherits from FEM and defines the interface for iterative solvers.
    """

    @abstractmethod
    def BForm(self):
        """
        Defines the bilinear form A(u, v).
        """
        pass

    
    @abstractmethod 
    def LForm(self):
        """
        Defines the linear form L(v).
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Updates the coefficients of matrices and vectors in each iteration step.
        """
        pass

    def lagrange_multiplier(self, A, b, c=0):
        """
        Constructs the augmented system matrix for Lagrange multipliers.
        c is the integral of pressure, default is 0.
        """

        LagLinearForm = LinearForm(self.pspace)
        LagLinearForm.add_integrator(SourceIntegrator(source=1))
        LagA = LagLinearForm.assembly()
        LagA = bm.concatenate([bm.zeros(self.uspace.number_of_global_dofs()), LagA], axis=0)

        A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
                                 bm.arange(len(LagA), dtype=bm.int32)]), LagA, spshape=(1, len(LagA)))

        A = BlockForm([[A, A1.T], [A1, None]])
        A = A.assembly_sparse_matrix(format='csr')
        b0 = bm.array([c])
        b  = bm.concatenate([b, b0], axis=0)
        return A, b
    
    @variantmethod('dirichlet') 
    def apply_bc(self, A, b, pde, t=None):
        """
        Apply dirichlet boundary conditions to velocity and pressure.
        """
        if t is None:
            BC = DirichletBC(
                (self.uspace, self.pspace), 
                gd=(pde.velocity_dirichlet, pde.pressure_dirichlet), 
                threshold=(pde.is_velocity_boundary, pde.is_pressure_boundary),
                method='interp')
            A, b = BC.apply(A, b)
        else:
            gd_v = cartesian(lambda p:pde.velocity_dirichlet(p, t))
            gd_p = cartesian(lambda p:pde.pressure_dirichlet(p, t))
            gd = (gd_v, gd_p)
            BC = DirichletBC(
                (self.uspace, self.pspace), 
                gd=gd, 
                threshold=(pde.is_velocity_boundary, pde.is_pressure_boundary),
                method='interp')
            A, b = BC.apply(A, b)
        return A, b
