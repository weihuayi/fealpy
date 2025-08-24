from typing import Any, Optional, Union

from fealpy.backend import bm
from fealpy.typing import TensorLike
from fealpy.decorator import variantmethod, cartesian
from fealpy.model import ComputationalModel
from fealpy.model.stokes import StokesPDEDataT

from fealpy.mesher import DLDMicrofluidicChipMesher
from fealpy.functionspace import functionspace 
from fealpy.fem import LinearForm, BilinearForm, BlockForm, LinearBlockForm
from fealpy.fem import ScalarDiffusionIntegrator as DiffusionIntegrator
from fealpy.fem import DirichletBC
from fealpy.fem import PressWorkIntegrator, SourceIntegrator


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
        set_inlet_condition: Set the PDE data for the model
        set_init_mesh: Set the initial mesh
        setup: Initialize PDE equation and FEM method
        set_space_degree: Set polynomial degree for function spaces
        linear_system: Assemble the linear system
        solve: Solve the linear system
    """
    def __init__(self, options: dict = None):
        self.options = options
        super().__init__(
            pbar_log=options['pbar_log'],
            log_level=options['log_level']
        )
        
    def set_init_mesher(self, mesher: DLDMicrofluidicChipMesher):
        """
        Set the initial mesh for the simulation.
        
        Parameters:
            mesh: The computational mesh object
        """
        self.mesh = mesher.mesh
        self.centers = mesher.centers
        self.boundary = mesher.boundary
        self.inlet_boundary = mesher.inlet_boundary
        self.outlet_boundary = mesher.outlet_boundary
        self.wall_boundary = mesher.wall_boundary
        self.project_edges = mesher.project_edges

    def set_space_degree(self, p: int=2):
        """
        Set the polynomial degree for function spaces
        """
        self.p = p

    def set_inlet_condition(self)-> None:
        """
        Set the PDE data for the model.
        """
        @cartesian
        def inlet_velocity(p: TensorLike) -> TensorLike:
            """Compute exact solution of velocity."""
            x = p[..., 0]
            y = p[..., 1]
            result = bm.zeros(p.shape, dtype=bm.float64)
            result[..., 0] = y*(1-y)
            result[..., 1] = bm.array(0.0)
            return result
        
        @cartesian
        def outlet_pressure(p: TensorLike) -> TensorLike:
            """Compute exact solution of velocity."""
            x = p[..., 0]
            y = p[..., 1]
            result = bm.zeros(p.shape[0], dtype=bm.float64)
            result[:] = 0.0
            return result
        
        @cartesian
        def is_inlet_boundary( p: TensorLike) -> TensorLike:
            """Check if point where velocity is defined is on boundary."""
            bd = self.inlet_boundary
            return self.is_boundary(p, bd)
        
        @cartesian
        def is_outlet_boundary( p: TensorLike) -> TensorLike:
            """Check if point where pressure is defined is on boundary."""
            bd = self.outlet_boundary
            return self.is_boundary(p, bd)

        @cartesian
        def is_wall_boundary(p: TensorLike) -> TensorLike:
            """Check if point where velocity is defined is on boundary."""
            bd = self.wall_boundary
            return self.is_boundary(p, bd)
        
        @cartesian
        def is_obstacle_boundary(p: TensorLike) -> TensorLike:
            """Check if point where velocity is defined is on boundary."""
            x = p[..., 0]
            y = p[..., 1]
            radius = self.options['radius']
            atol = 1e-12
            on_boundary = bm.zeros_like(x, dtype=bool)
            for center in self.centers:
                cx, cy = center
                on_boundary |= bm.abs((x - cx)**2 + (y - cy)**2 - radius**2) < atol
            return on_boundary
        
        self.inlet_velocity = inlet_velocity
        self.outlet_pressure = outlet_pressure
        self.is_inlet_boundary = is_inlet_boundary
        self.is_outlet_boundary = is_outlet_boundary
        self.is_wall_boundary = is_wall_boundary
        self.is_obstacle_boundary = is_obstacle_boundary

    def is_boundary(self, p: TensorLike, bd: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        atol = 1e-12
        v0 = p[:, None, :] - bd[None, 0::2, :] # (NN, NI, 2)
        v1 = p[:, None, :] - bd[None, 1::2, :] # (NN, NI, 2)

        cross = v0[..., 0]*v1[..., 1] - v0[..., 1]*v1[..., 0] # (NN, NI)
        dot = bm.einsum('ijk,ijk->ij', v0, v1) # (NN, NI)
        cond = (bm.abs(cross) < atol) & (dot < atol)
        return bm.any(cond, axis=1)
    
    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of pressure."""
        x = p[..., 0]
        y = p[..., 1]
        return 8*(1-x)
    
    @cartesian
    def is_velocity_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        inlet = self.is_inlet_boundary(p)
        wall = self.is_wall_boundary(p)
        obstacle = self.is_obstacle_boundary(p)
        return inlet | wall | obstacle
    
    @cartesian
    def is_pressure_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        return self.is_outlet_boundary(p)

    @cartesian
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
        """Optional: prescribed velocity on boundary, if needed explicitly."""
        inlet = self.inlet_velocity(p)
        is_inlet = self.is_inlet_boundary(p)
        
        result = bm.zeros_like(p, dtype=p.dtype)
        result[is_inlet] = inlet[is_inlet]
        
        return result
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        """Optional: prescribed pressure on boundary (usually for stability)."""
        outlet = self.outlet_pressure(p)
        is_outlet = self.is_outlet_boundary(p)

        result = bm.zeros_like(p[..., 0], dtype=p.dtype)
        result[is_outlet] = outlet[is_outlet]
        return result

    def linear_system(self):
        """
        Assemble the linear system for the Stokes equations.
        """
        GD = self.mesh.geo_dimension()

        self.uspace = functionspace(self.mesh, ('Lagrange', self.p), shape=(GD, -1))
        self.pspace = functionspace(self.mesh, ('Lagrange', self.p-1))

        A00 = BilinearForm(self.uspace)
        self.BD = DiffusionIntegrator()
        # self.BD.coef = 1.0
        A00.add_integrator(self.BD)
        A01 = BilinearForm((self.pspace, self.uspace))
        self.BP = PressWorkIntegrator()
        # self.BP.coef = -1.0
        A01.add_integrator(self.BP)
        A = BlockForm([[A00, A01], [A01.T, None]])

        L0 = LinearForm(self.uspace)
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

    @variantmethod('one_step')
    def run(self):
        """
        Run a single step of the simulation.
        """
        BForm, LForm = self.linear_system()
        A = BForm.assembly()
        L = LForm.assembly()    

        BC = DirichletBC(
            (self.uspace, self.pspace),
            gd=(self.velocity_dirichlet, self.pressure_dirichlet),
            threshold=(self.is_velocity_boundary, self.is_pressure_boundary),
            method='interp'
        )
        A, L = BC.apply(A, L)
        x = self.solve(A, L)
        ugdof = self.uspace.number_of_global_dofs()
        uh = x[:ugdof]
        ph = x[ugdof:-1]

        self.post_process(uh ,ph)
        return uh, ph
    
    def post_process(self, uh, ph):
        import matplotlib.pyplot as plt

        # uh = uh.reshape(2, -1).T 
        # points = self.uspace.interpolation_points()
        # fig, ax = plt.subplots(figsize=(8,6))
        # self.mesh.add_plot(ax)
        # ax.quiver(points[:,0], points[:,1], uh[:,0], uh[:,1])
        # ax.set_title("Velocity field")
        # plt.show()
        self.mesh.nodedata['ph'] = ph
        self.mesh.nodedata['uh'] = uh.reshape(2,-1).T
        self.mesh.to_vtk('dld_chip1.vtu')
    
