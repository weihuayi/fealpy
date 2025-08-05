from abc import ABC, abstractmethod

from fealpy.backend import backend_manager as bm
from fealpy.fem import LinearForm, SourceIntegrator, BlockForm
from fealpy.sparse import COOTensor
from fealpy.decorator import variantmethod
from fealpy.fem import DirichletBC

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
    
    def apply_bc(self, A, b, pde):
        """
        Apply dirichlet boundary conditions to velocity and pressure.
        """
        BC = DirichletBC(
            (self.uspace, self.pspace), 
            gd=(pde.velocity, pde.pressure), 
            threshold=(pde.is_velocity_boundary, pde.is_pressure_boundary),
            method='interp')
        A, b = BC.apply(A, b)
        return A, b
    ''' 
    @apply_bc.register("cylinder")
    def apply_bc(self, A, b, pde):
        """
        Apply boundary conditions for cylinder problems.
        """
        BC_influx = DirichletBC(
            (self.uspace, self.pspace), 
            gd=(pde.inlet_velocity, pde.pressure), 
            threshold=(pde.is_inlet_boundary, None),
            method='interp')
        
        BC_outflux = DirichletBC(
            (self.uspace, self.pspace),
            gd = (pde.outlet_velocity, pde.outlet_pressure),
            threshold=(pde.is_outlet_boundary, pde.is_outlet_boundary),
            method='interp')
        
        BC_wall = DirichletBC(
            (self.uspace, self.pspace), 
            gd=(pde.wall_velocity, pde.pressure), 
            threshold=(pde.is_wall_boundary, None),
            method='interp')
        
        BC_obstacle = DirichletBC(
            (self.uspace, self.pspace), 
            gd=(pde.obstacle_velocity, pde.pressure), 
            threshold=(pde.is_obstacle_boundary, None),
            method='interp')
        
        A, b = BC_influx.apply(A, b)
        A, b = BC_outflux.apply(A, b)
        A, b = BC_wall.apply(A, b)
        A, b = BC_obstacle.apply(A, b)
        self.apply_bc_str = "cylinder"
        return A, b
        '''
