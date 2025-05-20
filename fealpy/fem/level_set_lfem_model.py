import warnings

from ..backend import backend_manager as bm
from ..fem import ScalarMassIntegrator, ScalarConvectionIntegrator
from ..fem import ScalarDiffusionIntegrator
from ..fem import ScalarSourceIntegrator
from ..fem import BilinearForm, LinearForm
from ..solver import spsolve
from ..decorator import cartesian, barycentric
from ..functionspace import TensorFunctionSpace
from dataclasses import dataclass

@dataclass
class Options:
    def __init__(self):
        ## evolution Parameters
        self._evo_method: str = 'CN'
        self._evo_solver: str = 'mumps'
        self._evo_rein: bool = True
        
        ## reinitialization Parameters
        self._re_dt: float = 0.0001
        self._re_alpha: float = None
        self._re_space: int = 5
        self._re_solver: str = 'mumps'
        self._re_maxit: int = 1000
        self._re_tol: float = 5e-6
    
    @property
    def evo_method(self):
        return self._evo_method
    
    @property
    def evo_solver(self):
        return self._evo_solver

    @property
    def evo_rein(self):
        return self._evo_rein

    @property
    def re_dt(self):
        return self._re_dt

    @property
    def re_alpha(self):
        return self._re_alpha

    @property
    def re_space(self):
        return self._re_space

    @property
    def re_solver(self):
        return self._re_solver
    
    @property
    def re_maxit(self):
        return self._re_maxit

    @property
    def re_tol(self):
        return self._re_tol
    
    def set_evo_params(self, **kwargs):
        valid_params = {
            'evo_method': '_evo_method',
            'evo_solver': '_evo_solver',
            'evo_rein': '_evo_rein'
        }
        
        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, valid_params[key], value)
            else:
                raise ValueError(f"Unknown parameter: {key}")

    def set_reinit_params(self, **kwargs):
        valid_params = {
            're_dt': '_re_dt',
            're_alpha': '_re_alpha',
            're_space': '_re_space',
            're_solver': '_re_solver',
            're_maxit': '_re_maxit',
            're_tol': '_re_tol'
        }
        
        warnings.warn("Modifying reinitialization options may affect algorithm stability",
                     UserWarning)
        
        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, valid_params[key], value)
            else:
                raise ValueError(f"Unknown parameter: {key}")

class LevelSetLFEMModel():
    """
    A finite element solver for the level set evolution equation, which tracks
    the evolution of an interface driven by a velocity field and reinitialize 
    equation.     
    """
    def __init__(self, space, u, method:str = None, q:int = None):
        self.space = space
        self.options = Options()
        
        if q is None:
            self.q = space.p + 3
        else:
            self.q = q
        
        if method is None:
            self.options._evo_method = method

        
        self.u = u
        if not hasattr(self.u, 'coordtype') or self.u.coordtype == 'cartesian':
            uspace = TensorFunctionSpace(space, (space.mesh.GD,-1)) 
            self.u = uspace.interpolate(self.u) 
    

    def run(self, T, dt, phi0, output='./'):
        """
        Run the level set evolution solver.
        """
        from fealpy.solver import spsolve
        
        if not hasattr(phi0, 'coordtype') or phi0.coordtype == 'cartesian':
            phi0 = self.space.interpolate(phi0)
        
        nt = int(T/dt)
        Bform = self.Bform()
        Lform = self.Lform()
        self.output(phi0, self.u, 0, output)
        for i in range(nt):
            print("t=",i*dt)

            self.update(dt, phi0)
            A = Bform.assembly()
            b = Lform.assembly()
            
            x = spsolve(A, b, solver=self.options._evo_solver)
            phi0[:] = x
            if self.options._evo_rein:
                if i % self.options._re_space == 0:
                    phi0 = self.reinit_run(phi0)
            
            self.output(phi0, self.u, i, output)
        return phi0
    
    def linear_system(self, return_matrix=False):
        Bform = self.Bform()
        Lform = self.Lform()
        A = Bform.assembly()
        b = Lform.assembly()
        if return_matrix:
            return A, b
        else:
            return Bform, Lform


    def Bform(self, method=None):
        if method is None:
            method = self.options._evo_method
        
        if method == 'eular':
            return self.evolution_eular_Bfrom()
        elif method == 'CN':
            return self.evolution_CN_Bfrom()
        elif method == 'SUPG':
            return self.evolution_SUPG_Bfrom()
        else:
            raise ValueError('Invalid method')

    def Lform(self, method=None):
        if method is None:
            method = self.options._evo_method
        
        if method == 'eular':
            return self.evolution_eular_Lfrom()
        elif method == 'CN':
            return self.evolution_CN_Lfrom()
        elif method == 'SUPG':
            return self.evolution_SUPG_Lfrom()
        else:
            raise ValueError('Invalid method')

    def update(self, dt, phi, u=None, method=None):
        if method is None:
            method = self.options._evo_method
        
        if method == 'eular':
            self.evo_eular_update(dt, phi, u)
        elif method == 'CN':
            self.evo_CN_update(dt, phi, u)
        elif method == 'SUPG':
            self.evo_SUPG_update(dt, phi, u)
        else:
            raise ValueError('Invalid method')


    ### Crank-Nicolson method ###
    def evolution_CN_Lfrom(self):
        Lform = LinearForm(self.space)
        self.evo_CN_source = ScalarSourceIntegrator(q=self.q).keep_data(True)
        Lform.add_integrator(self.evo_CN_source)
        return Lform
    
    def evolution_CN_Bfrom(self):
        Bform = BilinearForm(self.space)
        Bform.add_integrator(ScalarMassIntegrator(q=self.q))
        self.evo_CN_con = ScalarConvectionIntegrator(q=self.q).keep_data(True)
        Bform.add_integrator(self.evo_CN_con)
        return Bform
    
    def evo_CN_update(self, dt, phi, u=None):
        if u is None:
            u = self.u
        else:
            if not hasattr(self.u, 'coordtype') or self.u.coordtype == 'cartesian':
                uspace = TensorFunctionSpace(self.space, (self.space.mesh.GD,-1)) 
                u = uspace.interpolate(self.u) 
        
        @barycentric
        def con_coef(bcs, index=None):
            result = u(bcs, index)
            return 0.5 * dt * result
        
        self.evo_CN_con.coef = con_coef
        #self.evo_CN_con.clear()
        
        @barycentric
        def source_coef(bcs, index=None):
            gradphi = phi.grad_value(bcs, index) 
            uu = u(bcs, index)
            result = phi(bcs, index) - 0.5 * dt * bm.einsum('...i, ...i -> ...', gradphi, uu)
            return result
        self.evo_CN_source.source = source_coef
        #self.evo_CN_source.clear()
    
    ### Eular method ###
    def evolution_eular_Lfrom(self):
        Lform = LinearForm(self.space)
        self.evo_eular_source = ScalarSourceIntegrator(q=self.q).keep_data(True)
        Lform.add_integrator(self.evo_eular_source)
        return Lform

    def evolution_eular_Bfrom(self):
        Bform = BilinearForm(self.space)
        Bform.add_integrator(ScalarMassIntegrator(q=self.q))
        self.evo_eular_con = ScalarConvectionIntegrator(q=self.q).keep_data(True)
        Bform.add_integrator(self.evo_eular_con)
        return Bform

    def evo_eular_update(self, dt, phi, u=None):
        if u is not None:
            if not hasattr(self.u, 'coordtype') or self.u.coordtype == 'cartesian':
                raise ValueError('The velocity field must be a finite element function(barycentric)') 
            self.u = u 

        @barycentric
        def con_coef(bcs, index=None):
            result = self.u(bcs, index)
            return dt * result
        
        self.evo_eular_con.coef = con_coef
        #self.evo_eular_con.clear()
        self.evo_eular_source.source = phi
        #self.evo_eular_source.clear()
    
    ### SUPG method ###
    
    ### Reinitialization ###
class LevelSetReinitModel():
    def __init__(self, phi0, q:int = None):
        
        self.space = phi0.space
        self.phi0 = phi0
        self.options = Options()
        if q is None:
            self.q = self.space.p + 3
        else:
            self.q = q
    
    def reinit_run(self):
        phi0 = self.phi0
        
        if self.options._re_alpha is None:
            cellscale = bm.max(self.space.mesh.entity_measure('cell'))
            self.options._re_alpha = 0.0625*cellscale 

        rephi0 = self.space.function()
        rephi1 = self.space.function()
        rephi0[:] = phi0[:]

        Bform = self.reinit_Bfrom()
        Lform = self.reinit_Lfrom()
        
        eold = 1e10

        for i in range(self.options._re_maxit):
            self.reinit_update(self.options._re_dt, rephi0)
            A = Bform.assembly()
            b = Lform.assembly()
            rephi1[:] = spsolve(A, b, solver=self.options._re_solver)

            error = self.space.mesh.error(rephi1, rephi0)
        
            if eold < error or error< self.options._re_tol:
                print("Reinitialization success")
                break
            else:
                rephi0[:] = rephi1[:]
                eold = error
        return rephi1


    def reinit_Lfrom(self):
        Lform = LinearForm(self.space)
        self.reinit_source = ScalarSourceIntegrator(q=self.q).keep_data(True)
        Lform.add_integrator(self.reinit_source)
        return Lform

    def reinit_Bfrom(self):
        Bform = BilinearForm(self.space)
        Bform.add_integrator(ScalarMassIntegrator(q=self.q))
        self.reinit_diff = ScalarDiffusionIntegrator(q=self.q).keep_data(True)
        Bform.add_integrator(self.reinit_diff)
        return Bform

    def reinit_update(self, dtau, rephi): 
        @barycentric
        def rein_source_coef(bcs, index=None):
            result = rephi(bcs, index)
            grad = rephi.grad_value(bcs, index)
            val = bm.linalg.norm(grad, axis=-1) - 1
            val *= bm.sign(self.phi0(bcs, index))
            result -= dtau * val 
            return result

        self.reinit_source.source = rein_source_coef
        #self.reinit_source.clear()
        self.reinit_diff.coef = self.options._re_alpha * dtau
        #self.reinit_diff.clear()

    ### Utility functions ###
    def output(self, phi, u, timestep, output):
        mesh = phi.space.mesh
        mesh.nodedata['phi'] = phi
        mesh.nodedata['velocity'] = u
        fname = output + str(timestep).zfill(10) + '.vtu'
        mesh.to_vtk(fname=fname)

    def check_gradient_norm_at_interface(self, phi, tolerance=1e-3):
        """
        Check the gradient magnitude of the level set function at the interface.

        Parameters:
        - phi: The level set function evaluated at quadrature points.
        - tolerance: The tolerance within which a point is considered part of the interface.

        Returns:
        - diff_avg: The average difference between the gradient magnitude and 1 at the interface.
        - diff_max: The maximum difference between the gradient magnitude and 1 at the interface.
        """
        # Compute phi and the gradient of phi at quadrature points
        mesh = phi.space.mesh
        space = phi.space
        qf = mesh.quadrature_formula(space.p+2)
        bcs, _ = qf.get_quadrature_points_and_weights()
        phi_quad = space.value(uh=phi, bc=bcs)
        grad_phi_quad = space.grad_value(uh=phi, bc=bcs)

        # Compute the magnitude of the gradient at quadrature points
        magnitude = bm.linalg.norm(grad_phi_quad, axis=-1)

        # Identify points at the interface
        at_interface_mask = bm.abs(phi_quad) <= tolerance

        # Compute the difference between the magnitude and 1 at the interface
        diff = bm.abs(magnitude[at_interface_mask]) - 1

        diff_avg = bm.mean(diff) if bm.any(at_interface_mask) else 0
        diff_max = bm.max(diff) if bm.any(at_interface_mask) else 0

        return diff_avg, diff_max

    def compute_zero_level_set_area(self, phi0):
        """
        Compute the area of the zero level set of the level set function.

        Parameters:
        - phi0: The level set function evaluated at grid points.

        Returns:
        - area: The computed area of the zero level set.
        """
        space = phi0.space
        mesh = space.mesh
        measure = space.function()
        measure[phi0[:] > 0] = 0
        measure[phi0[:] <= 0] = 1
        
        qf = mesh.quadrature_formula(space.p+2)
        bcs, ws = qf.get_quadrature_points_and_weights()
        cellmeasure = mesh.entity_measure('cell')
        
        area = bm.einsum('i, ji, j ->', ws, measure(bcs), cellmeasure)

        return area

    def level_x(self, phi, y):
        '''
        计算界面与水平直线y交点的x值
        '''
        ipoint = phi.space.interpolation_points()
        y_indices = bm.where(ipoint[:, 1]==y)[0]
        phi_y = phi[y_indices]
        sort_indeces = bm.argsort(bm.abs(phi_y))[:2]
        indices = y_indices[sort_indeces]
        if phi[indices[0]] < 1e-8:
            return ipoint[indices[0],0]
        else :
            zong = bm.abs(phi[indices[0]]) + bm.abs(phi[indices[1]])
            ws0 = 1 - bm.abs(phi[indices[0]])/zong
            ws1 = 1 - bm.abs(phi[indices[1]])/zong
            val = ws0 * ipoint[indices[0], 0] + ws1*ipoint[indices[1],0]
            return val

    class IterationCounter(object):
        def __init__(self, disp=True):
            self._disp = disp
            self.niter = 0

        def __call__(self, rk=None):
            self.niter += 1
            if self._disp:
                print('iter %3i' % (self.niter))
