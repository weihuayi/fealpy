import numpy as np
import time

from ..fem import BilinearForm
from ..fem import LinearForm
from ..fem import ScalarConvectionIntegrator
from ..fem import ScalarDiffusionIntegrator
from ..fem import ScalarSourceIntegrator
from ..fem import ScalarMassIntegrator

from ..decorator import barycentric

from .ls_solver import LSSolver

from scipy.sparse.linalg import spsolve


class LSFEMSolver(LSSolver):
    """
    A finite element solver for the level set evolution equation, which tracks
    the evolution of an interface driven by a velocity field. It discretizes
    the transport equation using Crank-Nicolson method in time and finite
    element method in space.
    """
    def __init__(self, space, u=None):
        """
        Initialize the finite element solver for the level set evolution.

        Parameters:
        - space : The finite element space over which the system is defined.
        - u : The velocity field driving the interface evolution. It should be a vector
            function defined on the mesh nodes.

        The bilinear forms for the mass and convection are assembled here. The
        mass matrix M represents the mass term in the equation, while the
        convection matrix C represents the velocity field's convection effect.
        """
        self.space = space

        # Assemble the mass matrix using the ScalarMassIntegrator which represents the L2 inner product of the finite element functions.
        bform = BilinearForm(space)
        bform.add_domain_integrator(ScalarMassIntegrator())
        self.M = bform.assembly() # TODO:Implement a fast assembly method

        self.u = u
        self.p = space.p

        # Assemble the convection matrix only if a velocity field is provided.
        if u is not None:
            bform = BilinearForm(space)

            # The convection integrator accounts for the transport effect of the velocity field on the level set function.
            bform.add_domain_integrator(ScalarConvectionIntegrator(c = u, q = 4))
            self.C = bform.assembly() # TODO:Implement a fast assembly method

    def lgmres_solve(self, phi0, dt, u=None, tol=1e-8):
        """
        Solve the level set evolution equation for one time step using the
        provided initial condition and velocity field.

        Parameters:
        - phi0 : The initial condition for the level set function.
        - dt : Time step size for the evolution.
        - u : (Optional) Updated velocity field for the evolution.
        - tol : Tolerance for the linear system solver.

        The function solves for phi^{n+1} given phi^n (phi0) using the
        discretized Crank-Nicolson scheme. It returns the updated level set
        function after one time step.
        """
        space = self.space
        M = self.M

        # Use the provided velocity field u for this time step if given, otherwise use the previously stored velocity field.
        if u is None:
            C = self.C 
            if C is None:
                raise ValueError(" Velocity `u` is None! You must offer velocity!")
        else:
            bform = BilinearForm(space)
            bform.add_domain_integrator(ScalarConvectionIntegrator(c = u, q = self.p+2))
            C = bform.assembly()

        # The system matrix A is composed of the mass matrix and the convection matrix.
        # It represents the Crank-Nicolson discretization of the PDE.
        A = M + (dt/2) * C 

        # The right-hand side vector b for the linear system includes the effect of the previous time step's level set function and the convection.
        b = M @ phi0 - (dt/2) * C @ phi0

        # Solve the linear system to find the updated level set function.
        phi0[:] = self.lgmres_solve_system(A, b, tol = tol)

        return phi0

    def mumps_solve(self, q, phi0, dt, u=None):
        """
        Solve the level set evolution equation for one time step using the
        provided initial condition and velocity field.

        Parameters:
        - phi0 : The initial condition for the level set function.
        - dt : Time step size for the evolution.
        - u : (Optional) Updated velocity field for the evolution.
        - tol : Tolerance for the linear system solver.

        The function solves for phi^{n+1} given phi^n (phi0) using the
        discretized Crank-Nicolson scheme. It returns the updated level set
        function after one time step.
        """
        space = self.space
        M = self.M

        # Use the provided velocity field u for this time step if given, otherwise use the previously stored velocity field.
        if u is None:
            C = self.C 
            if C is None:
                raise ValueError(" Velocity `u` is None! You must offer velocity!")
        else:
            bform = BilinearForm(space)
            bform.add_domain_integrator(ScalarConvectionIntegrator(c = u, q = q))
            C = bform.assembly()

        # The system matrix A is composed of the mass matrix and the convection matrix.
        # It represents the Crank-Nicolson discretization of the PDE.
        A = M + (dt/2) * C 

        # The right-hand side vector b for the linear system includes the effect of the previous time step's level set function and the convection.
        b = M @ phi0 - (dt/2) * C @ phi0

        # Solve the linear system to find the updated level set function.
        #phi0[:] = self.mumps_solve_system(A, b)
        result = self.mumps_solve_system(A, b)

        return result


    def solve_measure(self, phi0, dt, u=None, tol=1e-8):
        """
        Solve the level set evolution equation for one time step using the
        provided initial condition and velocity field.

        Parameters:
        - phi0 : The initial condition for the level set function.
        - dt : Time step size for the evolution.
        - u : (Optional) Updated velocity field for the evolution.
        - tol : Tolerance for the linear system solver.

        The function solves for phi^{n+1} given phi^n (phi0) using the
        discretized Crank-Nicolson scheme. It returns the updated level set
        function after one time step.
        """
        space = self.space
        start_time_M = time.time()
        M = self.M
        end_time_M = time.time()
        time_M = end_time_M - start_time_M

        start_time_C = time.time()
        # Use the provided velocity field u for this time step if given, otherwise use the previously stored velocity field.
        if u is None:
            C = self.C 
            if C is None:
                raise ValueError(" Velocity `u` is None! You must offer velocity!")
        else:
            bform = BilinearForm(space)
            bform.add_domain_integrator(ScalarConvectionIntegrator(c = u))
            C = bform.assembly()
        end_time_C = time.time()
        time_C = end_time_C - start_time_C

        # The system matrix A is composed of the mass matrix and the convection matrix.
        # It represents the Crank-Nicolson discretization of the PDE.
        A = M + (dt/2) * C 

        # The right-hand side vector b for the linear system includes the effect of the previous time step's level set function and the convection.
        b = M @ phi0 - (dt/2) * C @ phi0

        start_time_solve = time.time()
        # Solve the linear system to find the updated level set function.
        phi0 = self.solve_system(A, b, tol = tol)
        end_time_solve = time.time()
        time_solve = end_time_solve - start_time_solve

        print(f"Time to assemble M: {time_M} seconds")
        print(f"Time to assemble C: {time_C} seconds")
        print(f"Time to solve linear system: {time_solve} seconds")

        return phi0


    def reinit(self, phi0, dt = 0.0001, eps = 5e-6, nt = 4, alpha = None, show=False):
        '''
        Reinitialize the level set function to a signed distance function using the PDE-based reinitialization approach.

        This function solves a reinitialization equation to stabilize the level set
        function without moving the interface. The equation introduces artificial diffusion
        to stabilize the solution process.

        Parameters:
        - phi0: The level set function to be reinitialized.
        - dt: The timestep size for the pseudo-time evolution.
        - eps: A small positive value to avoid division by zero and to smooth the transition across the interface.
        - nt: The number of pseudo-time steps to perform.
        - alpha: The artificial diffusion coefficient, which is auto-calculated based on the cell scale if not provided.

        The reinitialization equation is discretized in time using a forward Euler method.
        The weak form of the reinitialization equation is assembled and solved iteratively
        until a stable state is reached.

        Returns:
        - phi1: The reinitialized level set function as a signed distance function.
        '''
        space = self.space
        mesh = space.mesh

        # Calculate the maximum cell scale which is used to determine the artificial diffusion coefficient.
        cellscale = np.max(mesh.entity_measure('cell'))
        if alpha is None:
            alpha = 0.0625*cellscale

        # Initialize the solution function and set its values to the initial level set function.
        phi1 = space.function()
        phi1[:] = phi0
        phi2 = space.function()

        # Assemble the mass matrix M.
        M = self.M

        # Assemble the stiffness matrix S using ScalarDiffusionIntegrator for artificial diffusion.
        bform = BilinearForm(space)
        bform.add_domain_integrator(ScalarDiffusionIntegrator())
        S = bform.assembly()

         # Initialize the old error.
        eold = 1e10

        # Iterate for a fixed number of pseudo-time steps or until the error is below a threshold.
        for _ in range(nt):

            # Define the function f for the right-hand side of the reinitialization equation.
            @barycentric
            def f(bcs, index):
                grad = phi1.grad_value(bcs)
                val = 1 - np.sqrt(np.sum(grad**2, -1))
                val *= np.sign(phi0(bcs))
                return val
            
            # Assemble the linear form for the right-hand side of the equation.
            lform = LinearForm(space)
            lform.add_domain_integrator( ScalarSourceIntegrator(f = f) )
            b0 = lform.assembly()

             # Compute the right-hand side vector for the linear system.
            b = M @ phi1 + dt * b0 - dt * alpha * (S @ phi1)

            # Solve the linear system to update the level set function.
            phi2[:] = spsolve(M, b)

            # Calculate the error between the new and old level set function.
            error = space.mesh.error(phi2, phi1)
            if show == True:
                print("Reinitialization error:", error) 

            # If the error starts increasing or is below the threshold, break the loop.
            if eold < error or error< eps :
                print("Reinitialization success")
                break
            else:
                phi1[:] = phi2
                eold = error

        return phi1

