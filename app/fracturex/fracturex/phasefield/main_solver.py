from typing import Optional
from scipy.sparse import spdiags
from fealpy.experimental.sparse import SparseTensor, COOTensor, CSRTensor

from fealpy.experimental.typing import TensorLike
from fealpy.experimental.backend import backend_manager as bm

from fealpy.experimental.decorator import barycentric, cartesian

from fealpy.experimental.fem import BilinearForm, LinearForm
from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.experimental.fem import LinearElasticIntegrator
from fealpy.experimental.fem import ScalarDiffusionIntegrator, ScalarMassIntegrator
from fealpy.experimental.fem import ScalarSourceIntegrator
from fealpy.experimental.solver import cg

from app.fracturex.fracturex.phasefield.energy_degradation_function import EnergyDegradationFunction as EDFunc
from app.fracturex.fracturex.phasefield.phase_fracture_material import PhaseFractureMaterialFactory




class MainSolver:
    def __init__(self, mesh, material_params, p=1, q=None, method='HybridModel'):
        """
        Parameters
        ----------
        material_params : material properties 'lam' : lame params, 'mu' : shear modulus , 'E' : Young's modulus, 'nu' : Poisson's ratio, 'Gc' : critical energy release rate, 'l0' : length scale
        method : str
            The method of stress decomposition method
        """
        self.mesh = mesh
        self.p = p
        self.q = self.p+2 if q is None else q

        self.EDFunc = EDFunc()
        self.pfcm = PhaseFractureMaterialFactory.create('HybridModel', material_params, self.EDFunc)
        
        self.set_lfe_space()
        self.Gc = material_params['Gc']
        self.l0 = material_params['l0']
        
        self.bc_dict = {}

    def solve(self):
        """
        
        Solve the phase field fracture problem.

        parameters
        ----------
        disp_increment_boundary : function
            A function that returns the displacement increment boundary condition of the displacement field.

        inc_value : float
            The value of the displacement increment.
        """
        force_type = self.get_boundary_conditions('force')['type']
        force_dof = self.get_boundary_conditions('force')['bcdof']
        force_value = self.get_boundary_conditions('force')['value']
        force_direction = self.get_boundary_conditions('force')['direction']
        self.force_dof = force_dof
        self.force_direction = force_direction

        if force_type is not 'Dirichlet':
            raise ValueError('Unsupported boundary condition type')
        
        fbc = DirichletBC(self.tspace, gd=0, threshold=force_dof, direction=force_direction)

        for i in range(len(force_value)):
            self.uh, self.force_index = fbc.apply_value(self.uh)
            print('uh', self.uh)
            #self.solve_displacement(force)
            #self.solve_phase_field()
        
        

    def newton_raphson(self):
        """
        Newton-Raphson iteration
        """
        pass


    def solve_displacement(self):
        """
        Solve the displacement field.
        """
        uh = self.uh
        ubform = BilinearForm(self.tspace)
        ubform.add_integrator(LinearElasticIntegrator(self.pfcm, q=self.q))
        A = ubform.assembly()
        R = A@uh[:]

        self.force = bm.sum(-R[self.force_index]) # 计算残余力的大小
        
        if self.bc_dict['displacement'] or self.bc_dict['both'] is not None:
            bc_type= self.get_boundary_conditions('displacement')['type']
            disp_bc_dof = self.get_boundary_conditions('displacement')['bcdof']
            disp_bc_value = self.get_boundary_conditions('displacement')['value']
            direction = self.get_boundary_conditions('displacement')['direction']
            
            if bc_type == 'Dirichlet':
                 ubc = DirichletBC(self.tspace, gd=disp_bc_value, threshold=disp_bc_dof, direction=direction)
                 A, R = ubc.apply(A, R)
            else:
                raise ValueError('Unsupported boundary condition type')
        
        ubc = DirichletBC(self.tspace, 0, threshold=self.force_dof, direction=self.force_direction)  
        A, R = ubc.apply(A, R)
        du = cg(A, R, atol=1e-14)[0]
        uh += du.flat[:]

        self.pfcm.update_disp(uh)
        

    def solve_phase_field(self):
        """
        Solve the phase field.
        """
    
        @barycentric
        def coef(bc, index):
            return 2*self.pfcm.maximum_historical_field(bc)
        
        Gc = self.Gc
        l0 = self.l0
        d = self.d

        dbfrom = BilinearForm(self.space)
        dbfrom.add_integrator(ScalarDiffusionIntegrator(Gc*l0, q=self.q))
        dbfrom.add_integrator(ScalarMassIntegrator(Gc/l0, q=self.q))
        dbfrom.add_integrator(ScalarMassIntegrator(coef, q=self.q))
        A = dbfrom.assembly()

        dlfrom = LinearForm(self.space)
        dlfrom.add_integrator(ScalarSourceIntegrator(coef, q=self.q))
        R = dlfrom.assembly()
        R -= A@d[:]


    def set_lfe_space(self):
        """
        Set the function space for the problem.
        """
        self.space = LagrangeFESpace(self.mesh, self.p)
        self.tspace = TensorFunctionSpace(self.space, (self.mesh.geo_dimension(), -1))

        self.d = self.space.function()
        self.uh = self.tspace.function()

        self.pfcm.update_disp(self.uh)
        self.pfcm.update_phase(self.d)
        
    
    def vtkfile(self):
        pass


    def add_boundary_condition(self, field_type: str, bc_type: str, boundary_dof: Optional[TensorLike], value: Optional[TensorLike], direction: Optional[str] = None):
        """
        Add boundary condition for a specific field and boundary.

        Parameters
        ----------
        field_type : str
            'froce', 'displacement' or 'phase_field'.
        
        bc_type : str
            Type of the boundary condition ('Dirichlet', 'Neumann', etc.).
        boundary_dof : [TensorLike]
            dof of the boundary.
        value : Optional[TensorLike]
            The value of the boundary condition.
        """
        if field_type not in self.bc_dict:
            self.bc_dict[field_type] = {}
        self.bc_dict[field_type] = {'type': bc_type, 'bcdof': boundary_dof,  'value': value, 'direction': direction}

    def get_boundary_conditions(self, field_type: str):
        """
        Get the boundary conditions for a specific field.

        Parameters
        ----------
        field_type : str
            'froce', 'displacement' or 'phase_field'.
        """
        return self.bc_dict.get(field_type, {})

class DirichletBC:
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
            index[:] = is_bd_dof  # Apply to all directions
        else:
            idx = direction_map.get(direction)
            if idx is not None and idx < GD:
                index[idx] = is_bd_dof  # Apply only to the specified direction
            else:
                raise ValueError(f"Invalid direction '{direction}' for GD={GD}. Use 'x', 'y', 'z', or None.")
    
        # Flatten the index to return as a 1D array
        return index.ravel()

    def apply_value(self, u):
        index = self.set_boundary_dof()
        u[index] = self.gd

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
        index = self.set_boundary_dof()
        kwargs = A.values_context()

        if isinstance(A, COOTensor):
            indices = A.indices()
            remove_flag = bm.logical_or(
                index[indices[0, :]], index[indices[1, :]]
            )
            retain_flag = bm.logical_not(remove_flag)
            new_indices = indices[:, retain_flag]
            new_values = A.values()[..., retain_flag]
            A = COOTensor(new_indices, new_values, A.sparse_shape)

            index = bm.nonzero(index)[0]
            shape = new_values.shape[:-1] + (len(index), )
            one_values = bm.ones(shape, **kwargs)
            one_indices = bm.stack([index, index], axis=0)
            A1 = COOTensor(one_indices, one_values, A.sparse_shape)
            A = A.add(A1).coalesce()

        elif isinstance(A, CSRTensor):
            raise NotImplementedError('The CSRTensor version has not been implemented.')
        return A, f


