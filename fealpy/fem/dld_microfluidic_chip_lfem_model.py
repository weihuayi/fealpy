from typing import Any, Optional, Union

from fealpy.backend import bm
from fealpy.typing import TensorLike
from fealpy.decorator import variantmethod
from fealpy.model import ComputationalModel
from fealpy.model.stokes import StokesPDEDataT

from fealpy.functionspace import functionspace 
from fealpy.fem import LinearForm, BilinearForm, BlockForm, LinearBlockForm
from fealpy.fem import ScalarDiffusionIntegrator as DiffusionIntegrator
from fealpy.fem import DirichletBC
from fealpy.fem import PressWorkIntegrator, SourceIntegrator

from fealpy.cfd.model import CFDTestModelManager
from fealpy.cfd.equation.stationary_stokes import StationaryStokes


class DLDMicrofluidicChipLFEMModel(ComputationalModel):
    """
    A Lagrange finite element computational model class for Deterministic Lateral 
    Displacement (DLD) microfluidic chip simulation.

    Parameters:
        options (dict, optional): A dictionary containing computational options 
            for the model. Expected keys include:
            - pbar_log: Whether to enable progress bar logging
            - log_level: Logging level for the model

    Attributes:
        pde (StokesPDEDataT): The PDE data object containing problem definition
        mesh: The computational mesh
        equation (StationaryStokes): The Stokes equation object
        fem: The finite element method implementation
        uspace: Velocity function space
        pspace: Pressure function space
        p (int): Polynomial degree for function spaces

    Methods:
        set_pde: Set the PDE data for the model
        set_init_mesh: Set the initial mesh
        setup: Initialize PDE equation and FEM method
        method: Configure the FEM method variant
        set_space_degree: Set polynomial degree for function spaces
        linear_system: Assemble the linear system
        solve: Solve the linear system
        update_mesh: Update the computational mesh
        update: Update equation coefficients
        run: Run the simulation
        error: Compute solution errors
    """
    def __init__(self, options: dict = None):
        self.options = options
        super().__init__(
            pbar_log=options['pbar_log'],
            log_level=options['log_level']
        )

    def set_pde(self, pde=1)-> None:
        """
        Set the PDE data for the model.
        """
        if isinstance(pde, int):
            self.pde = CFDTestModelManager('stationary_stokes').get_example(pde)
        else:
            self.pde = pde

    def set_init_mesh(self, mesh):
        """
        Set the initial mesh for the simulation.
        
        Parameters:
            mesh: The computational mesh object
        """
        self.mesh = mesh
    
    def setup(self):
        """
        Initialize the PDE-related equation object and the FEM method.
        """
        if self.pde is None:
            raise ValueError("PDE not set. Call set_pde(pde) first.")
        if self.mesh is None:
            raise ValueError("Mesh not set. Call set_mesh(mesh) first.")
        self.equation = StationaryStokes(self.pde)
        self.fem = self.method()

    @variantmethod("Stokes")
    def method(self): 
        """
        Use Newton iteration method to solve the Navier-Stokes equations.
        """
        from ..cfd.simulation.fem.stationary_stokes.stokes import Stokes
        self.fem = Stokes(self.equation, self.mesh)
        self.method_str = "Stokes"
        return self.fem

    def set_space_degree(self, p: int=2):
        """
        Set the polynomial degree for function spaces
        """
        self.p = p

    def linear_system(self):
        """
        Assemble the linear system for the Stokes equations.
        """
        GD = self.mesh.geo_dimension()
        self.uspace = functionspace(self.mesh, ('Lagrange', self.p), shape=(GD, -1))
        self.pspace = functionspace(self.mesh, ('Lagrange', self.p-1))

        A00 = BilinearForm(self.uspace)
        self.u_BVW = DiffusionIntegrator()
        A00.add_integrator(self.u_BVW)
        
        A01 = BilinearForm((self.pspace, self.uspace))
        self.u_BPW = PressWorkIntegrator()
        A01.add_integrator(self.u_BPW)
       
        A = BlockForm([[A00, A01], [A01.T, None]]) 

        L0 = LinearForm(self.uspace)
        self.u_source_LSI = SourceIntegrator()
        L0.add_integrator(self.u_source_LSI)

        L1 = LinearForm(self.pspace)
        L = LinearBlockForm([L0, L1])

        return A, L

    @variantmethod('direct')
    def solve(self, A, F, solver='scipy'):
        """
        Solve the linear system using direct method.
        """
        from fealpy.solver import spsolve
        self.solve_str = 'direct'
        return spsolve(A, F, solver = solver)
    
    def update_mesh(self, mesh):
        """
        Update the mesh used in the model.
        """
        self.mesh = mesh
        self.fem.update_mesh(mesh)

    def update(self): 
        """
        Update equation coefficients from the current PDE state.
        """
        equation = self.equation
        cv = equation.coef_viscosity
        pc = equation.coef_pressure
        cbf = equation.coef_body_force
        
        ## BilinearForm
        self.u_BVW.coef = cv
        self.u_BPW.coef = -pc

        ## LinearForm 
        self.u_source_LSI.source = cbf

    @variantmethod('one_step')
    def run(self):
        """
        Run a single step of the simulation.
        """
        self.run_str = 'one_step'
        BForm, LForm = self.linear_system()  
        self.update()
        A = BForm.assembly()
        b = LForm.assembly()
        A, b = self.fem.apply_bc(A, b, self.pde)
        A, b = self.fem.lagrange_multiplier(A, b)
        x = self.solve(A, b)

        ugdof = self.uspace.number_of_global_dofs()
        u = self.uspace.function()
        p = self.pspace.function()
        # u = bm.set_at(u, slice(None), x[:ugdof])
        # p = bm.set_at(p, slice(None), x[ugdof:-1])
        u[:] = x[:ugdof]
        p[:] = x[ugdof:-1] 
        return u, p

    @run.register('uniform_refine')
    def run(self, maxit = 3):
        """
        Run simulation with uniform mesh refinement.
        """
        self.run_str = 'uniform_refine'
        u_errorMatrix = bm.zeros((1, maxit), dtype=bm.float64)
        p_errorMatrix = bm.zeros((1, maxit), dtype=bm.float64)
        for i in range(maxit):
            self.logger.info(f"number of cells: {self.mesh.number_of_cells()}")
            uh, ph = self.run['one_step']()
            uerror, perror = self.error(uh, ph)
            u_errorMatrix[0, i] = uerror
            p_errorMatrix[0, i] = perror
            order_u = bm.log2(u_errorMatrix[0,:-1]/u_errorMatrix[0,1:])
            order_p = bm.log2(p_errorMatrix[0,:-1]/p_errorMatrix[0,1:])
            self.mesh.uniform_refine()
        self.logger.info(f"速度最终误差:" + ",".join(f"{uerror:.15e}" for uerror in u_errorMatrix[0,]))
        self.logger.info(f"order_u: " + ", ".join(f"{order_u:.15e}" for order_u in order_u))
        self.logger.info(f"压力最终误差:" + ",".join(f"{perror:.15e}" for perror in p_errorMatrix[0,]))  
        self.logger.info(f"order_p: " + ", ".join(f"{order_p:.15e}" for order_p in order_p))
        return uh, ph

    def error(self, uh, ph):
        """
        Post-process the numerical solution to compute the error in L2 norm.
        """
        self.error_str = 'error'
        uerror = self.mesh.error(self.pde.velocity, uh)
        perror = self.mesh.error(self.pde.pressure, ph)
        return uerror, perror
