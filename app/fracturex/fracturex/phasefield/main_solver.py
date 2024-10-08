from typing import Optional, Dict
import numpy as np
from fealpy.utils import timer
from scipy.sparse import spdiags

from fealpy.experimental.typing import TensorLike
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.decorator import barycentric, cartesian
from fealpy.experimental.fem import BilinearForm, LinearForm
from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.experimental.fem import LinearElasticIntegrator, ScalarDiffusionIntegrator, ScalarMassIntegrator, ScalarSourceIntegrator
from fealpy.experimental.fem import DirichletBC
from fealpy.experimental.solver import cg

from app.fracturex.fracturex.phasefield.energy_degradation_function import EnergyDegradationFunction as EDFunc
from app.fracturex.fracturex.phasefield.phase_fracture_material import PhaseFractureMaterialFactory
from app.fracturex.fracturex.phasefield.adaptive_refinement import AdaptiveRefinement
from app.fracturex.fracturex.phasefield.vector_Dirichlet_bc import VectorDirichletBC


class MainSolver:
    def __init__(self, mesh, material_params: Dict, 
                 p: int = 1, q: Optional[int] = None, 
                 method: str = 'HybridModel',
                 enable_refinement: bool = False, 
                 marking_strategy: str = 'recovery', 
                 refine_method: str = 'bisect', 
                 theta: float = 0.2):
        """
        Initialize the MainSolver class with more customization options.

        Parameters
        ----------
        mesh : object
            The mesh for the problem.
        material_params : dict
            Dictionary containing material properties: 'lam', 'mu', 'E', 'nu', 'Gc', 'l0'.
        p : int, optional
            Polynomial order for the function space, by default 1.
        q : int, optional
            Quadrature degree, by default p + 2.
        method : str, optional
            Stress decomposition method, by default 'HybridModel'.
        enable_refinement : bool, optional
            Whether to enable adaptive refinement, by default False.
        marking_strategy : str, optional
            The marking strategy for refinement, by default 'recovery'.
        refine_method : str, optional
            The refinement method, by default 'nvp'.
        theta : float, optional
            Refinement threshold parameter, by default 0.2.
        """
        self.mesh = mesh
        self.p = p
        self.q = self.p + 2 if q is None else q
        self.method = method

        # Material and energy degradation function
        self.EDFunc = EDFunc()
        self.pfcm = PhaseFractureMaterialFactory.create(method, material_params, self.EDFunc)

        # Initialize spaces
        self.set_lfe_space()
        self.pfcm.update_disp(self.uh)
        self.pfcm.update_phase(self.d)

        # Material parameters
        self.Gc = material_params['Gc']
        self.l0 = material_params['l0']
        
        self.bc_dict = {}

        self.enable_refinement = False

        # Initialize the timer
        self.tmr = timer()
        next(self.tmr)


    def solve(self, maxit: int = 50, save_vtkfile : bool = True, vtkname: Optional[str] = None):
        """
        Solve the phase-field fracture problem.

        Parameters
        ----------
        maxit : int, optional
            Maximum number of iterations, by default 30.
        vtkname : str, optional
            VTK output file name, by default None.
        """
        self._initialize_force_boundary()
        self.Rforce = bm.zeros_like(self.force_value)
        
        for i in range(len(self.force_value)-1):
            print('i', i)
            self._currt_force_value = self.force_value[i+1]
            # Run Newton-Raphson iteration
            self.newton_raphson(maxit)
            
            if save_vtkfile:
                if vtkname is None:
                    fname = f'test{i:010d}.vtu'
                else:
                    fname = f'{vtkname}{i:010d}.vtu'
                self.save_vtkfile(fname=fname)
            
            bm.set_at(self.Rforce, i+1, self.Rfu)
            
            

    def newton_raphson(self, maxit: int = 50):
        """
        Perform the Newton-Raphson iteration for solving the problem.

        Parameters
        ----------
        maxit : int, optional
            Maximum number of iterations, by default 30.
        force_value : TensorLike
            Value of the force boundary condition.
        """
        tmr = self.tmr
        for k in range(maxit):
            print(f"Newton-Raphson Iteration {k + 1}/{maxit}:")
            
            tmr.send('start')
            er0 = self.solve_displacement()
            self.pfcm.update_disp(self.uh)
            tmr.send('disp_solve')
            
            print(f"Displacement error after iteration {k + 1}: {er0}")

            er1 = self.solve_phase_field()
            self.pfcm.update_phase(self.d)
            print(f"Phase field error after iteration {k + 1}: {er1}")
            tmr.send('phase_solve')

            self.H = self.pfcm.H
            if self.enable_refinement:
                data = self.set_interpolation_data()
                self.mesh, new_data = self.adaptive.perform_refinement(self.mesh, self.d, data, self.l0)
                if new_data:
                    self.set_lfe_space()
                    self.update_interpolation_data(new_data)
                    print(f"Refinement after iteration {k + 1}")

            tmr.send('refine')
            if k == 0:
                e0, e1 = er0, er1
            
            error = max(er0/e0, er1/e1)
            tmr.send('end')
            print(f'Iteration {k}, Error: {error}')
            if error < 1e-5:
                print(f"Convergence achieved after {k + 1} iterations.")
                break

    def solve_displacement(self) -> float:
        """
        Solve the displacement field and return the residual norm.

        Returns
        -------
        float
            The norm of the residual.
        """
        uh = self.uh
        tmr = self.tmr
        tmr.send('start')

        fbc = VectorDirichletBC(self.tspace, self._currt_force_value, self.force_dof, direction=self.force_direction)
        uh, force_index = fbc.apply_value(uh)
        self.pfcm.update_disp(uh)

        ubform = BilinearForm(self.tspace)
        ubform.add_integrator(LinearElasticIntegrator(self.pfcm, q=self.q))
        A = ubform.assembly()
        R = -A @ uh[:]
        self.Rfu = bm.sum(-R[force_index])
        tmr.send('disp_assemble')

        # Apply force boundary conditions
        ubc = VectorDirichletBC(self.tspace, 0, threshold=self.force_dof, direction=self.force_direction)  
        A, R = ubc.apply(A, R)

        # Apply displacement boundary conditions
        A, R = self._apply_boundary_conditions(A, R, field='displacement')
        tmr.send('apply_bc')

        du = cg(A.tocsr(), R, atol=1e-14)
        uh += du.flatten()[:]
        self.uh = uh
        
        tmr.send('disp_solve')
        return np.linalg.norm(R)

    def solve_phase_field(self) -> float:
        """
        Solve the phase field and return the residual norm.

        Returns
        -------
        float
            The norm of the residual.
        """
        Gc, l0, d = self.Gc, self.l0, self.d
        @barycentric
        def coef(bc, index):
            return 2 * self.pfcm.maximum_historical_field(bc)

        tmr = self.tmr
        tmr.send('start')

        dbform = BilinearForm(self.space)
        dbform.add_integrator(ScalarDiffusionIntegrator(Gc * l0, q=self.q, method='fast'))
        dbform.add_integrator(ScalarMassIntegrator(Gc / l0, q=self.q))
        dbform.add_integrator(ScalarMassIntegrator(coef, q=self.q))
        A = dbform.assembly()
        tmr.send('phase_matrix_assemble')

        dlform = LinearForm(self.space)
        dlform.add_integrator(ScalarSourceIntegrator(coef, q=self.q))
        R = dlform.assembly()
        R -= A @ d[:]
        tmr.send('phase_R_assemble')

        A, R = self._apply_boundary_conditions(A, R, field='phase')
        tmr.send('phase_apply_bc')

        dd = cg(A.tocsr(), R, atol=1e-14)
        d += dd.flatten()[:]

        self.d = d
        tmr.send('phase_solve')
        return np.linalg.norm(R)

    def set_lfe_space(self):
        """
        Set the finite element spaces for displacement and phase fields.
        """
        self.space = LagrangeFESpace(self.mesh, self.p)
        self.tspace = TensorFunctionSpace(self.space, (self.mesh.geo_dimension(), -1))
        self.d = self.space.function()
        self.uh = self.tspace.function()

    def set_adaptive_refinement(self, marking_strategy: str = 'recovery', refine_method: str = 'bisect', theta: float = 0.2):
        """
        Set the adaptive refinement parameters.
        ----------
        marking_strategy : str, optional
            The marking strategy for refinement, by default 'recovery'.
        refine_method : str, optional
            The refinement method, by default 'bisect'.
        theta : float, optional
            Mark threshold parameter, by default 0.2.        
        """
        # Adaptive refinement settings
        self.enable_refinement = True
        self.adaptive = AdaptiveRefinement(marking_strategy=marking_strategy, refine_method=refine_method, theta=theta)

    def set_interpolation_data(self):
        """
        Set the interpolation data to refine.
        """
        GD = self.mesh.geo_dimension()
        #NQ = self.H.shape[-1] if len(self.H) > 1 else 1
        if GD == 2:
            dcell2dof = self.space.cell_to_dof()
            ucell2dof = self.tspace.cell_to_dof()
            data = {'uh': self.uh[ucell2dof], 'd': self.d[dcell2dof], 'H': self.H}
        elif GD == 3:
            data = {'nodedata':[self.uh, self.d], 'celldata':self.H}
        return data

    def update_interpolation_data(self, data):
        """
        Update the data after refinement
        """
        GD = self.mesh.geo_dimension()
        if GD == 2:
            dcell2dof = self.space.cell_to_dof()
            ucell2dof = self.tspace.cell_to_dof()
            self.uh[ucell2dof.reshape(-1)] = data['uh']
            self.d[dcell2dof.reshape(-1)] = data['d']
            H = data['H']
        elif GD == 3:
            assert self.p == 1
            self.uh = data['uh']
            self.d = data['d']

            H = data['H']
        self.H = H
        self.pfcm.update_historical_field(self.H)
        self.pfcm.update_disp(self.uh)
        self.pfcm.update_phase(self.d)


    def save_vtkfile(self, fname: str):
        """
        Save the solution to a VTK file.

        Parameters
        ----------
        fname : str
            File name for saving the VTK output.
        """
        mesh = self.mesh
        GD = mesh.geo_dimension()
        mesh.nodedata['damage'] = self.d
        mesh.nodedata['uh'] = self.uh.reshape(GD, -1).T
        mesh.to_vtk(fname=fname)

    def add_boundary_condition(self, field_type: str, bc_type: str, boundary_dof: TensorLike, value: TensorLike, direction: Optional[str] = None):
        """
        Add boundary condition for a specific field and boundary.

        Parameters
        ----------
        field_type : str
            'force', 'displacement', or 'phase'.
        bc_type : str
            Type of the boundary condition ('Dirichlet', 'Neumann', etc.).
        boundary_dof : TensorLike
            DOF of the boundary.
        value : TensorLike
            Value of the boundary condition.
        direction : str, optional
            Direction for vector fields ('x', 'y', 'z'), by default None.
        """
        self.bc_dict[field_type] = {'type': bc_type, 'bcdof': boundary_dof, 'value': value, 'direction': direction}

    def get_boundary_conditions(self, field_type: str):
        """
        Get the boundary conditions for a specific field.

        Parameters
        ----------
        field_type : str
            'force', 'displacement', or 'phase'.

        Returns
        -------
        dict
            Boundary condition data for the specified field.
        """
        return self.bc_dict.get(field_type, {})

    def _initialize_force_boundary(self):
        """
        Initialize the force boundary conditions.
        """
        force_data = self.get_boundary_conditions('force')
        self.force_type = force_data.get('type')
        self.force_dof = force_data.get('bcdof')
        self.force_value = force_data.get('value')
        self.force_direction = force_data.get('direction')

    def _apply_boundary_conditions(self, A, R, field: str):
        """
        Apply boundary conditions to the system matrix and residual.

        Parameters
        ----------
        A : sparse matrix
            System matrix.
        R : ndarray
            Residual vector.
        field : str
            Field type: 'displacement' or 'phase'.
        """
        bcdata = self.get_boundary_conditions(field)
    
        if bcdata:
            if field == 'displacement':
                if bcdata['type'] == 'Dirichlet':
                    bc = VectorDirichletBC(self.tspace, bcdata['value'], bcdata['bcdof'], direction=bcdata['direction'])
                    A, R = bc.apply(A, R)
                else:
                    raise NotImplementedError(f"Boundary condition '{bcdata['type']}' is not implemented.")
            elif field == 'phase':
                if bcdata['type'] == 'Dirichlet':
                    bc = DirichletBC(self.space, bcdata['value'], bcdata['bcdof'])
                    A, R = bc.apply(A, R)
                else:
                    raise NotImplementedError(f"Boundary condition '{bcdata['type']}' is not implemented.")
        return A, R

