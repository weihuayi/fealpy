from typing import Optional
import numpy as np
from fealpy.utils import timer

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
from fealpy.experimental.fem import DirichletBC

from fealpy.experimental.solver import cg


from app.fracturex.fracturex.phasefield.energy_degradation_function import EnergyDegradationFunction as EDFunc
from app.fracturex.fracturex.phasefield.phase_fracture_material import PhaseFractureMaterialFactory

#import matplotlib.pyplot as plt


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
        self.pfcm.update_disp(self.uh)
        self.pfcm.update_phase(self.d)

        self.Gc = material_params['Gc']
        self.l0 = material_params['l0']
        
        self.bc_dict = {}
        self.tmr = timer()
        next(self.tmr)

    def solve(self, maxit=30, vtkname=None):
        """
        
        Solve the phase field fracture problem.

        parameters
        ----------
        maxit : int
            The maximum number of iterations.
        vtkname : str
            The name of the vtk file.
        """
        tmr = self.tmr
        force_type = self.get_boundary_conditions('force')['type']
        force_dof = self.get_boundary_conditions('force')['bcdof']
        force_value = self.get_boundary_conditions('force')['value']
        force_direction = self.get_boundary_conditions('force')['direction']
        self.force_dof = force_dof
        self.force_direction = force_direction

        if force_type != 'Dirichlet':
            raise ValueError('Unsupported boundary condition type')
        
        self.Rforce = bm.zeros_like(force_value)
        tmr.send('init')
        for i in range(2):
            print('i', i)
            fbc = VectorDirichletBC(self.tspace, gd=force_value[i+1], threshold=force_dof, direction=force_direction)

            self.uh, force_index = fbc.apply_value(self.uh) # Apply the displacement condition
            self.pfcm.update_disp(self.uh)
            print('uh', self.uh)
            
            self.newton_raphson(maxit) # Newton-Raphson iteration
            tmr.send('solve')

            if vtkname is None:
                fname = 'test' + str(i).zfill(10)  + '.vtu'
            fname = vtkname + str(i).zfill(10)  + '.vtu'
            self.save_vtkfile(fname=fname)
            
            bm.set_at(self.Rforce, i+1, bm.sum(self.Ru[force_index])) # Calculate the size of the residual force
        tmr.send('stop')
        next(tmr)
        #self.draw_force_displacement_curve(force_value, Rforce)
        

    def newton_raphson(self, maxit=30):
        """
        Newton-Raphson iteration
        """
        tmr = self.tmr
        k = 0
        while k < maxit:
            er0 = self.solve_displacement()
            self.pfcm.update_disp(self.uh)
            tmr.send('solve_displacement')
            print('uh', self.uh)

            er1 = self.solve_phase_field()
            self.pfcm.update_phase(self.d)
            tmr.send('solve_phase_field')
            print('d', self.d)

            self.H = self.pfcm._H
            print('H', self.H)

            if k == 0:
                e0 = er0
                e1 = er1
            error = max(er0/e0, er1/e1)
            print('k:', k, 'error:', error)
            if error < 1e-5:
                break
            k += 1
            tmr.send('stop')
            next(tmr)

    def solve_displacement(self):
        """
        Solve the displacement field.
        """
        tmr = self.tmr
        uh = self.uh
        ubform = BilinearForm(self.tspace)
        ubform.add_integrator(LinearElasticIntegrator(self.pfcm, q=self.q))
        A = ubform.assembly()
        R = -A@uh[:]
        self.Ru = R[:]
        
        tmr.send('disp_assembly')

        if self.bc_dict.get('displacement') is not None or self.bc_dict.get('both') is not None:
            bc_key = 'displacement' if self.bc_dict.get('displacement') is not None else 'both'
    
            # Extract the boundary condition details using the appropriate key
            bc_data = self.get_boundary_conditions(bc_key)
            bc_type= bc_data['type']
            disp_bc_dof = bc_data['bcdof']
            disp_bc_value = bc_data['value']
            direction = bc_data['direction']
            
            if bc_type == 'Dirichlet':
                 ubc = VectorDirichletBC(self.tspace, gd=disp_bc_value, threshold=disp_bc_dof, direction=direction)
                 A, R = ubc.apply(A, R)
            else:
                raise ValueError('Unsupported boundary condition type')
        
        ubc = VectorDirichletBC(self.tspace, 0, threshold=self.force_dof, direction=self.force_direction)  
        A, R = ubc.apply(A, R)
        tmr.send('disp_bc')

        du = cg(A, R, atol=1e-14)

        uh[:] += du.flat[:]

        tmr.send('disp_solve')
        self.uh = uh
        
        return np.linalg.norm(R)
        

    def solve_phase_field(self):
        """
        Solve the phase field.
        """
        tmr = self.tmr
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
        tmr.send('phase_martix_assembly')

        dlfrom = LinearForm(self.space)
        dlfrom.add_integrator(ScalarSourceIntegrator(coef, q=self.q))
        R = dlfrom.assembly()
        R -= A@d[:]
        tmr.send('phase_source_assembly')

        if self.bc_dict.get('phase') is not None or self.bc_dict.get('both') is not None:
            bc_key = 'phase' if self.bc_dict.get('phase') is not None else 'both'
    
            # Extract the boundary condition details using the appropriate key
            bc_data = self.get_boundary_conditions(bc_key)
            bc_type= bc_data['type']
            phase_bc_dof = bc_data['bcdof']
            phase_bc_value = bc_data['value']

            
            if bc_type == 'Dirichlet':
                 dbc = DirichletBC(self.space, gd=phase_bc_value, threshold=phase_bc_dof)
                 A, R = dbc.apply(A, R)
            else:
                raise ValueError('Unsupported boundary condition type')
        tmr.send('phase_bc')
        dd = cg(A, R, atol=1e-14)

        d += dd.flat[:]
        tmr.send('phase_solve')
        
        self.d = d
        
        return np.linalg.norm(R)

    def set_lfe_space(self):
        """
        Set the function space for the problem.
        """
        self.space = LagrangeFESpace(self.mesh, self.p)
        self.tspace = TensorFunctionSpace(self.space, (self.mesh.geo_dimension(), -1))

        self.d = self.space.function()
        self.uh = self.tspace.function()


        
    
    def save_vtkfile(self, fname):
        """
        Save the solution to a vtk file.
        """

        mesh = self.mesh
        GD = mesh.geo_dimension()

        mesh.nodedata['damage'] = self.d
        mesh.nodedata['uh'] = self.uh.reshape(GD, -1).T
        
        mesh.to_vtk(fname=fname)

    '''
    def draw_force_displacement_curve(self, disp, force):
        """
        Draw the force-displacement curve.
        """
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(disp, force, label='Force')
        plt.xlabel('disp')
        plt.ylabel('Force')
        plt.grid(True)
        plt.legend()
        plt.savefig('force.png', dpi=300)
    '''
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


