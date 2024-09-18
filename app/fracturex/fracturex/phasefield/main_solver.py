from typing import Optional, Dict
import numpy as np
from fealpy.utils import timer
from scipy.sparse import spdiags
from fealpy.experimental.sparse import SparseTensor, COOTensor, CSRTensor
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


class MainSolver:
    def __init__(self, mesh, material_params: Dict, p: int = 1, q: Optional[int] = None, method: str = 'HybridModel'):
        """
        Initialize the MainSolver class.

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
        self.tmr = timer()
        next(self.tmr)

    def solve(self, maxit: int = 30, vtkname: Optional[str] = None):
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
        
        for i in range(1):
            fbc = VectorDirichletBC(self.tspace, self.force_value[i+1], self.force_dof, direction=self.force_direction)
            self.uh, force_index = fbc.apply_value(self.uh)
            self.pfcm.update_disp(self.uh)

            # Run Newton-Raphson iteration
            self.newton_raphson(maxit)
            self.H = self.pfcm._H
            
            if vtkname is None:
                fname = f'test{i:010d}.vtu'
            else:
                fname = f'{vtkname}{i:010d}.vtu'
            self.save_vtkfile(fname=fname)

            bm.set_at(self.Rforce, i+1, bm.sum(self.Ru[force_index]))

    def newton_raphson(self, maxit: int = 30):
        """
        Perform the Newton-Raphson iteration for solving the problem.

        Parameters
        ----------
        maxit : int, optional
            Maximum number of iterations, by default 30.
        """
        for k in range(maxit):
            print('uh', self.uh)
            er0 = self.solve_displacement()
            self.pfcm.update_disp(self.uh)
            print('uh', self.uh)

            er1 = self.solve_phase_field()
            self.pfcm.update_phase(self.d)
            print('d', self.d)

            error = max(er0, er1) if k == 0 else max(er0/self.e0, er1/self.e1)
            if k == 0:
                self.e0, self.e1 = er0, er1

            print(f'Iteration {k}, Error: {error}')
            if error < 1e-5:
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
        ubform = BilinearForm(self.tspace)
        ubform.add_integrator(LinearElasticIntegrator(self.pfcm, q=self.q))
        A = ubform.assembly()
        R = -A @ uh[:]
        self.Ru = R[:]

        # Apply force boundary conditions
        ubc = VectorDirichletBC(self.tspace, 0, threshold=self.force_dof, direction=self.force_direction)  
        A, R = ubc.apply(A, R)
        
        # Apply displacement boundary conditions
        A, R = self._apply_boundary_conditions(A, R, field='displacement')
        du = cg(A, R, atol=1e-14)
        uh[:] += du.flat[:]
        self.uh = uh
        
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

        dbform = BilinearForm(self.space)
        dbform.add_integrator(ScalarDiffusionIntegrator(Gc * l0, q=self.q))
        dbform.add_integrator(ScalarMassIntegrator(Gc / l0, q=self.q))
        dbform.add_integrator(ScalarMassIntegrator(coef, q=self.q))
        A = dbform.assembly()

        dlform = LinearForm(self.space)
        dlform.add_integrator(ScalarSourceIntegrator(coef, q=self.q))
        R = dlform.assembly()
        R -= A @ d[:]

        A, R = self._apply_boundary_conditions(A, R, field='phase')
        dd = cg(A, R, atol=1e-14)
        d += dd.flat[:]
        self.d = d
        
        return np.linalg.norm(R)

    def set_lfe_space(self):
        """
        Set the finite element spaces for displacement and phase fields.
        """
        self.space = LagrangeFESpace(self.mesh, self.p)
        self.tspace = TensorFunctionSpace(self.space, (self.mesh.geo_dimension(), -1))
        self.d = self.space.function()
        self.uh = self.tspace.function()

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

class VectorDirichletBC:
    def __init__(self, space, gd, threshold, direction = None):
        self.space = space
        self.gd = gd
        self.threshold = threshold
        self.direction = direction

    def set_boundary_dof(self):
        threshold = self.threshold  # Boundary detection function
        direction = self.direction   # Direction for applying boundary condition
        space = self.space
        mesh = space.mesh
        ipoints = mesh.interpolation_points(p=space.p)  # Interpolation points
        is_bd_dof = threshold(ipoints)  # Boolean mask for boundary DOFs
        GD = mesh.geo_dimension()  # Geometric dimension (2D or 3D)

        # Prepare an index array with shape (GD, npoints)
        index = bm.zeros((GD, ipoints.shape[0]), dtype=bool)

        # Map direction to axis: 'x' -> 0, 'y' -> 1, 'z' -> 2 (for GD = 3)
        direction_map = {'x': 0, 'y': 1, 'z': 2}

        if direction is None:
            bm.set_at(index, slice(None), is_bd_dof)  # Apply to all directions
        else:
            idx = direction_map.get(direction)
            if idx is not None and idx < GD:
                bm.set_at(index, idx, is_bd_dof)  # Apply only to the specified direction
            else:
                raise ValueError(f"Invalid direction '{direction}' for GD={GD}. Use 'x', 'y', 'z', or None.")
    
        # Flatten the index to return as a 1D array
        return index.ravel()

    def apply_value(self, u):
        index = self.set_boundary_dof()
        bm.set_at(u, index, self.gd) 
        return u, index   

    def apply(self, A, f, u=None):
        """
        Apply Dirichlet boundary condition to the linear system.

        Parameters
        ----------
        A : SparseTensor
            The coefficient matrix.
        f : TensorLike
            The right-hand-side vector.

        Returns
        -------
        A : SparseTensor
            The new coefficient matrix.
        f : TensorLike
            The new right-hand-side vector.
        """
        isDDof = self.set_boundary_dof()
        kwargs = A.values_context()

        if isinstance(A, COOTensor):
            indices = A.indices()
            remove_flag = bm.logical_or(
                isDDof[indices[0, :]], isDDof[indices[1, :]]
            )
            retain_flag = bm.logical_not(remove_flag)
            new_indices = indices[:, retain_flag]
            new_values = A.values()[..., retain_flag]
            A = COOTensor(new_indices, new_values, A.sparse_shape)

            index = bm.nonzero(isDDof)[0]
            shape = new_values.shape[:-1] + (len(index), )
            one_values = bm.ones(shape, **kwargs)
            one_indices = bm.stack([index, index], axis=0)
            A1 = COOTensor(one_indices, one_values, A.sparse_shape)
            A = A.add(A1).coalesce()

        elif isinstance(A, CSRTensor):
            raise NotImplementedError('The CSRTensor version has not been implemented.')

        bm.set_at(f, isDDof, self.gd) 
        return A, f


