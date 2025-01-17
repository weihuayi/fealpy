from typing import Optional, Dict
import numpy as np
from fealpy.utils import timer

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.decorator import barycentric
from fealpy.fem import BilinearForm, LinearForm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import LinearElasticIntegrator, ScalarDiffusionIntegrator, ScalarMassIntegrator, ScalarSourceIntegrator

# 自动微分模块
from fealpy.fem import NonlinearForm
from fealpy.fem import ScalarNonlinearMassIntegrator, ScalarNonlinearDiffusionIntegrator
from fealpy.fem import NonlinearElasticIntegrator

# 边界处理模块
from fealpy.fem import DirichletBC

from fealpy.solver import cg, spsolve, gmres
from scipy.sparse.linalg import lgmres

from app.fracturex.fracturex.phasefield.energy_degradation_function import EnergyDegradationFunction as EDFunc
from app.fracturex.fracturex.phasefield.crack_surface_density_function import CrackSurfaceDensityFunction as CSDFunc
from app.fracturex.fracturex.phasefield.phase_fracture_material import PhaseFractureMaterialFactory
from app.fracturex.fracturex.phasefield.adaptive_refinement import AdaptiveRefinement
from app.fracturex.fracturex.phasefield.vector_Dirichlet_bc import VectorDirichletBC


class MainSolve:
    def __init__(self, mesh, material_params: Dict, 
                 model_type: str = 'HybridModel'):
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
            Quadrature degree, by default p + 3.
        model_type : str, optional
            Stress decomposition model, by default 'HybridModel'.
        method : str, optional
            The method for solving the problem, by default 'lfem'.
        """
        self.mesh = mesh

        self.model_type = model_type

        self.material_params = material_params

        # Material parameters
        self.Gc = material_params['Gc']
        self.l0 = material_params['l0']
        
        self.bc_dict = {}

        self.CSDFunc = None
        self.EDFunc = None

        self.enable_refinement = False

        self._save_vtk = False
        self._atype = None
        self._timer = False

        self._solver = None

        # Initialize the timer
        self.tmr = timer()
        next(self.tmr)
    
    def initialize_settings(self, p: int = 1, q: int = None, ):
        """
        Initialize the settings for the problem.
        """
        # Material and energy degradation function
        if self.EDFunc is None:
            self.set_energy_degradation(degradation_type='quadratic')
        if self.CSDFunc is None:
            self.set_crack_surface_density(density_type='AT2')

        self.pfcm = PhaseFractureMaterialFactory.create(self.model_type, self.material_params, self.EDFunc)

        # Initialize spaces
        if self._method == 'lfem':
            self.set_lfe_space(p=p, q=q)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.pfcm.update_disp(self.uh)
        self.pfcm.update_phase(self.d)

        # solver
        if self._solver is None:                
            if bm.device_type(self.uh) == 'cpu':
                print('Using scipy solver.')
                self._solver = 'scipy'
            elif bm.device_type(self.uh) == 'cuda':
                print('Using cupy solver.')
                self._solver = 'cupy'
            else:
                raise ValueError(f"Unknown device type: {bm.device_type(self.uh)}")

    def solve(self, method: str = 'lfem', p: int = 1, q: int = None, maxit: int = 50):
        """
        Solve the phase-field fracture problem.

        Parameters
        ----------
        maxit : int, optional
            Maximum number of iterations, by default 30.
        atype : str, optional
            Type of the solver, by default None. if 'auto', using automatic differentiation to assemble matrix. 
        vtkname : str, optional
            VTK output file name, by default None.
        """
        self._method = method
        self.initialize_settings(p=p, q=q)
        self._initialize_force_boundary()
        self._Rforce = bm.zeros_like(self._force_value)
        
        #for i in range(2):
        for i in range(len(self._force_value)-1):
            print('i', i)
            self._currt_force_value = self._force_value[i+1]

            # Run Newton-Raphson iteration
            self.newton_raphson(maxit)
            
            if self._save_vtk:
                if self._vtkfname is None:
                    fname = f'test{i:010d}.vtu'
                else:
                    fname = f'{self._vtkfname}{i:010d}.vtu'
                self._save_vtkfile(fname=fname)
            
            bm.set_at(self._Rforce, i+1, self._Rfu)
            

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

            # Solve the displacement field
            if self._method == 'lfem':
                if self._atype == 'auto':
                    er0 = self.solve_displacement_auto()
                else:
                    er0 = self.solve_displacement()
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Solve the phase field
            if self._method == 'lfem':
                if self._atype == 'auto':
                    print(f"Using automatic differentiation to assemble phase field matrix.")
                    er1 = self.solve_phase_field_auto()
                else:
                    er1 = self.solve_phase_field()
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Adaptive refinement
            if self.enable_refinement:
                data = self.set_interpolation_data()
                self.mesh, new_data = self.adaptive.perform_refinement(self.mesh, self.d, data, self.l0)
                if new_data:
                    if self.method == 'lfem':
                        self.set_lfe_space()
                    else:
                        raise ValueError(f"Unknown method: {self.method}")
                    self.update_interpolation_data(new_data)
                    print(f"Refinement after iteration {k + 1}")

                tmr.send('refine')

            # Check for convergence
            if k == 0:
                e0, e1 = er0, er1
            
            error = max(er0/e0, er1/e1)
            if self._timer:
                tmr.send(None)

            print(f"Displacement error after iteration {k + 1}: {er0/e0}")
            print(f"Phase field error after iteration {k + 1}: {er1/e1}")
            print(f'Iteration {k+1}, Error: {error}')
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
        tmr.send('disp_start')

        fbc = VectorDirichletBC(self.tspace, self._currt_force_value, self._force_dof, direction=self._force_direction)
        uh, force_index = fbc.apply_value(uh)
        self.pfcm.update_disp(uh)

        ubform = BilinearForm(self.tspace)
        ubform.add_integrator(LinearElasticIntegrator(self.pfcm, q=self.q, method='voigt'))
        A = ubform.assembly()
        R = -A @ uh[:]
        self._Rfu = bm.sum(-R[force_index])
        tmr.send('disp_assemble')

        # Apply force boundary conditions
        ubc = VectorDirichletBC(self.tspace, 0, threshold=self._force_dof, direction=self._force_direction)  
        A, R = ubc.apply(A, R)

        # Apply displacement boundary conditions
        A, R = self._apply_boundary_conditions(A, R, field='displacement')
        tmr.send('apply_bc')
        
        du = self.solver(A, R, atol=1e-20)
        uh += du[:]
        self.uh = uh
        
        self.pfcm.update_disp(uh)
        tmr.send('disp_solver')
        return bm.linalg.norm(R)

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
        def diff_coef(bc, index):
            gg_hd, c_d = self.CSDFunc.grad_grad_density_function(d(bc))
            return Gc * l0 * 2 / c_d

        @barycentric
        def mass_coef1(bc, index):
            gg_hd, c_d = self.CSDFunc.grad_grad_density_function(d(bc))
            return gg_hd * Gc / (l0 * c_d)
        
        
        @barycentric
        def mass_coef2(bc, index):
            gg_gd = self.EDFunc.grad_grad_degradation_function(d(bc))
            self.H = self.pfcm.maximum_historical_field(bc)
            return gg_gd * self.H
        
        @barycentric
        def source_coef(bc, index):
            gc_gd = self.EDFunc.grad_degradation_function_constant_coef()
            return -1 * gc_gd * self.H

        tmr = self.tmr
        tmr.send('phase_start')

        dbform = BilinearForm(self.space)
        dbform.add_integrator(ScalarDiffusionIntegrator(coef=diff_coef, q=self.q), ScalarMassIntegrator(coef=mass_coef1, q=self.q), ScalarMassIntegrator(coef=mass_coef2, q=self.q))
        A = dbform.assembly()
        tmr.send('phase_matrix_assemble')

        dlform = LinearForm(self.space)
        dlform.add_integrator(ScalarSourceIntegrator(source=source_coef, q=self.q))
        R = dlform.assembly()
        R -= A @ d[:] 
 
        
        tmr.send('phase_R_assemble')

        A, R = self._apply_boundary_conditions(A, R, field='phase')
        tmr.send('phase_apply_bc')
        
        dd = self.solver(A, R, atol=1e-20)
        d += dd[:]
  

        self.d = d
        self.pfcm.update_phase(d)
        self.H = self.pfcm.H

        tmr.send('phase_solver')
        return bm.linalg.norm(R)
    
    def solve_displacement_auto(self) -> float:
        """
        Using automatic differentiation to assemble matrix and solve the displacement field and return the residual norm.

        Returns
        -------
        float
            The norm of the residual.
        """
        uh = self.uh
        tmr = self.tmr
        tmr.send('diap_start')

        fbc = VectorDirichletBC(self.tspace, self._currt_force_value, self._force_dof, direction=self._force_direction)
        uh, force_index = fbc.apply_value(uh)
        self.pfcm.update_disp(uh)

        @barycentric
        def postive_coef(bc, **kwargs):
            return self.EDFunc.degradation_function(self.d(bc))
        
        postive_coef.uh = uh
        postive_coef.kernel_func = self.pfcm.positive_stress_func

        ubform = NonlinearForm(self.tspace)

        if self.model_type == 'HybridModel' or self.model_type == 'IsotropicModel':
            ubform.add_integrator(NonlinearElasticIntegrator(coef=postive_coef, material=self.pfcm, q=self.q))
        else:
            @barycentric
            def negative_coef(bc, **kwargs):
                return 1
            
            negative_coef.uh = uh
            negative_coef.kernel_func = self.pfcm.negative_stress_func

            ubform.add_integrator(NonlinearElasticIntegrator(coef=postive_coef, material=self.pfcm, q=self.q), NonlinearElasticIntegrator(coef=negative_coef, material=self.pfcm, q=self.q))
            
        A, R = ubform.assembly()

        self._Rfu = bm.sum(-R[force_index])
        tmr.send('disp_assemble')

        # Apply force boundary conditions
        ubc = VectorDirichletBC(self.tspace, 0, threshold=self._force_dof, direction=self._force_direction)  
        A, R = ubc.apply(A, R)

        # Apply displacement boundary conditions
        A, R = self._apply_boundary_conditions(A, R, field='displacement')
        tmr.send('apply_bc')

        du = self.solver(A, R, atol=1e-20)

        uh += du[:]
        self.uh = uh
        
        self.pfcm.update_disp(uh)
        tmr.send('disp_solver')
        return bm.linalg.norm(R)
    
    def solve_phase_field_auto(self) -> float:
        """
        Solve the phase field and return the residual norm.

        Returns
        -------
        float
            The norm of the residual.
        """
        Gc, l0, d = self.Gc, self.l0, self.d
        
        c_d = self.CSDFunc.grad_density_function(d)[1]
        @barycentric
        def diffusion_coef(bc, **kwargs):
            return Gc * l0 * 2 / c_d
        
        def diffusion_kernel_func(u):
            return u
        
        def diffusion_grad_kernel_func(u):
            return 1

        @barycentric
        def mass_coef1(bc, **kwargs):
            return Gc / (l0 * c_d)
        
        @barycentric
        def mass_kernel_func1(u):
            return self.CSDFunc.grad_density_function(u)[0]

        def mass_grad_kernel_func1(u):
            return self.CSDFunc.grad_grad_density_function(u)[0]
                
        @barycentric
        def mass_coef2(bc, **kwargs):
            self.H = self.pfcm.maximum_historical_field(bc)
            return self.H
        
        @barycentric
        def mass_kernel_func2(u):
            return self.EDFunc.grad_degradation_function(u)
        
        @barycentric
        def mass_grad_kernel_func2(u):
            return self.EDFunc.grad_grad_degradation_function(u)
        
        diffusion_coef.kernel_func = diffusion_kernel_func
        mass_coef1.kernel_func = mass_kernel_func1
        mass_coef2.kernel_func = mass_kernel_func2

        if bm.backend_name == 'numpy':
            diffusion_coef.grad_kernel_func = diffusion_grad_kernel_func
            mass_coef1.grad_kernel_func = mass_grad_kernel_func1
            mass_coef2.grad_kernel_func = mass_grad_kernel_func2

        mass_coef1.uh = d
        mass_coef2.uh = d
        diffusion_coef.uh = d

        tmr = self.tmr
        tmr.send('phase_start')

        # using automatic differentiation to assemble the phase field system        
        dform = NonlinearForm(self.space)
        dform.add_integrator(ScalarNonlinearDiffusionIntegrator(diffusion_coef, q=self.q), ScalarNonlinearMassIntegrator(mass_coef1, q=self.q), ScalarNonlinearMassIntegrator(mass_coef2, q=self.q)) 
        #dform.add_integrator(ScalarNonlinearMassIntegrator(mass_coef1, q=self.q))
        #dform.add_integrator(ScalarNonlinearMassIntegrator(mass_coef2, q=self.q))
        #dform.add_integrator(ScalarSourceIntegrator(source_coef, q=self.q))

        A, R = dform.assembly()
        tmr.send('phase_matrix_assemble')

        A, R = self._apply_boundary_conditions(A, R, field='phase')
        tmr.send('phase_apply_bc')

        dd = self.solver(A, R, atol=1e-20)
        d += dd[:]

        self.d = d
        self.pfcm.update_phase(d)
        self.H = self.pfcm.H

        tmr.send('phase_solver')
        return bm.linalg.norm(R)


    def set_lfe_space(self, p: int = 1, q: int = None):
        """
        Set the finite element spaces for displacement and phase fields.
        """
        self.p = p
        self.q = self.p + 3 if q is None else q
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

    def _save_vtkfile(self, fname: str):
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

    def save_vtkfile(self, fname: str):
        """
        Save the solution to a VTK file.

        Parameters
        ----------
        fname : str
            File name for saving the VTK output.
        """
        self._vtkfname = fname
        self._save_vtk = True

    def auto_assembly_matrix(self):
        """
        Assemble the system matrix using automatic differentiation.
        """
        assert bm.backend_name != "numpy", "In the numpy backend, you cannot use automatic differentiation method to assembly matrix."
        self._atype = 'auto'

    def fast_assembly_matrix(self):
        """
        Assemble the system matrix using fast assembly.
        """
        self._atype = 'fast'

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
        if field_type not in self.bc_dict:
            self.bc_dict[field_type] = []
    
        self.bc_dict[field_type].append({
            'type': bc_type,
            'bcdof': boundary_dof,
            'value': value,
            'direction': direction
        })

        
    def _get_boundary_conditions(self, field_type: str):
        """
        Get the boundary conditions for a specific field.

        Parameters
        ----------
        field_type : str
            'force', 'displacement', or 'phase'.

        Returns
        -------
        list
            A list of boundary condition data for the specified field.
        """
        return self.bc_dict.get(field_type, [])


    def _initialize_force_boundary(self):
        """
        Initialize the force boundary conditions.
        """
        force_data = self._get_boundary_conditions('force')
        force_data = force_data[0]
        self._force_type = force_data.get('type')
        self._force_dof = force_data.get('bcdof')
        self._force_value = force_data.get('value')
        self._force_direction = force_data.get('direction')

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
        bc_list = self._get_boundary_conditions(field)

        if bc_list:
            for bcdata in bc_list:
                if field == 'displacement':
                    if bcdata['type'] == 'Dirichlet':
                        bc = VectorDirichletBC(self.tspace, bcdata['value'], bcdata['bcdof'], direction=bcdata['direction'])
                        A, R = bc.apply(A, R)
                    else:
                        raise NotImplementedError(f"Boundary condition '{bcdata['type']}' is not implemented.")
                elif field == 'phase':
                    if bcdata['type'] == 'Dirichlet':
                        bc = DirichletBC(self.space, gd=bcdata['value'], threshold=bcdata['bcdof'])
                        A, R = bc.apply(A, R)
                    else:
                        raise NotImplementedError(f"Boundary condition '{bcdata['type']}' is not implemented.")
        return A, R
    
    def get_residual_force(self):
        """
        Get the residual vector.
        """
        return self._Rforce
    
    def output_timer(self):
        self._timer = True


    def set_energy_degradation(self, degradation_type='quadratic', EDfunc=None, **kwargs):
        """
        Set the energy degradation function.

        Parameters
        ----------
        EDfunc : callable, optional
            Energy degradation function class or factory. If None, a default energy degradation function is used.
        degradation_type : str, optional
            Type of energy degradation function. Default is 'quadratic'.
        **kwargs : dict
            Additional parameters passed to the energy degradation function.
        """
        if EDfunc is not None:
            self.EDFunc = EDfunc(degradation_type='user_defined', **kwargs)
        else:
            self.EDFunc = EDFunc(degradation_type=degradation_type)

    def set_crack_surface_density(self, density_type='AT2', CSDfunc=None, **kwargs):
        """
        Set the crack surface density function.

        Parameters
        ----------
        CSDFunc : callable, optional
            Crack surface density function class or factory. If None, a default crack surface density function is used.
        density_type : str, optional
            Type of crack surface density function. Default is 'AT2'.
        **kwargs : dict
            Additional parameters passed to the crack surface density function.
        """
        if CSDfunc is not None:
            self.CSDFunc = CSDfunc(density_type=='user_defined', **kwargs)
        else:
            self.CSDFunc = CSDFunc(density_type=density_type)

    def set_cupy_solver(self):
        """
        Use GPU to solve the problem.
        """
        print('Using cupy solver.')
        self._solver = 'cupy'

    def set_scipy_solver(self):
        """
        Use GPU to solve the problem.
        """
        print('Using scipy solver.')
        self._solver = 'scipy'

    def solver(self, A, R, atol=1e-20):
        """
        Choose the solver.
        """
        if self._solver == 'scipy':
            A = A.to_scipy()
            R = bm.to_numpy(R)

            x,info = lgmres(A, R, atol=atol)
            x = bm.tensor(x)
        elif self._solver == 'cupy':
            x = gmres(A, R, atol=atol, solver=self._solver)
        else:
            raise ValueError(f"Unknown solver: {self._solver}")
        return x
        
