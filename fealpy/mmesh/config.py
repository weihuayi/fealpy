from ..backend import backend_manager as bm
from ..typing import TensorLike
from ..mesh import TriangleMesh,TetrahedronMesh,QuadrangleMesh,HexahedronMesh
from ..mesh import LagrangeTriangleMesh,LagrangeQuadrangleMesh
from ..functionspace import LagrangeFESpace,ParametricLagrangeFESpace,Function
from ..fem import ( BilinearForm 
                    ,LinearForm
                    ,ScalarDiffusionIntegrator
                    ,ScalarMassIntegrator
                    ,ScalarSourceIntegrator
                    ,ScalarConvectionIntegrator
                    ,DirichletBC)
from ..solver import spsolve,cg
from ..sparse import CSRTensor,COOTensor,spdiags
from ..sparse.ops import bmat,hstack,vstack
from scipy.integrate import solve_ivp
from typing import Any ,Union,Optional

_U =  Union[TriangleMesh,
      TetrahedronMesh,
      QuadrangleMesh,
      HexahedronMesh,
      LagrangeTriangleMesh,
      LagrangeQuadrangleMesh]

_V = Union[LagrangeFESpace, ParametricLagrangeFESpace]

__all__ = [
    'bm', 'TensorLike', 
    'TriangleMesh', 'TetrahedronMesh', 'QuadrangleMesh', 'HexahedronMesh',
    'LagrangeTriangleMesh', 'LagrangeQuadrangleMesh',
    'LagrangeFESpace', 'ParametricLagrangeFESpace',
    'BilinearForm', 'LinearForm', 
    'ScalarDiffusionIntegrator', 
    'ScalarSourceIntegrator',
    'ScalarConvectionIntegrator',
    'ScalarMassIntegrator', 
    'DirichletBC', 
    'spsolve', 'cg', 
    'CSRTensor', 'COOTensor', 'spdiags','bmat', 'hstack', 'vstack',
    'solve_ivp', 'Any', 'Union', 'Optional','Function',
    '_U',
    '_V', 
    'Config'
]

class Config:
    def __init__(self,
                 beta:Union[float, list] = 0.5,
                 r:Union[float, list] = 0.15,
                 alpha = 0.5,
                 kappa = 0.5,
                 mol_times = 1,
                 active_method:str = 'Harmap',
                 pde = None,
                 logic_domain = None,
                 tol = None,
                 tau = None,
                 t_max = 0.5,
                 maxit = 500,
                 is_pre = True,
                 pre_steps = 5,
                 fun_solver:callable = None,
                 monitor:str = 'arc_length',
                 mol_meth:str = 'heatequ',
                 int_meth:str = 'comass'):
        """
        @param r: parameter of the mollification method
        @param alpha: parameter of the mollification method
        @param mol_times: times of the mollification
        @param method: method of moving mesh
        @param logic_domain: logic domain of the problem
        @param pde: PDEData instance
        @param tol: tolerance of the solver
        @param maxit: maximum iteration of the solver
        @param is_pre: whether to use preprocessor
        @param pre_steps: number of preprocessor steps
        @param fun_solver: function solver
        @param is_multi_phy: whether multi-physics problem
        @param monitor: monitor function
        @param mol_meth: mollification method
        @param int_meth: interpolation method
        @param odes_solver: odes solver
        """
        self.beta = beta
        self.r = r
        self.alpha = alpha
        self.mol_times = mol_times
        self.active_method = active_method
        self.pde = pde
        self.logic_domain = logic_domain
        self.tol = tol
        self.maxit = maxit
        self.is_pre = is_pre
        self.pre_steps = pre_steps
        self.tau = tau
        self.t_max = t_max
        self.fun_solver = fun_solver
        self.monitor = monitor
        self.mol_meth = mol_meth
        self.int_meth = int_meth
        self.parallel_mode = 'none'
        self.kappa = kappa
        self._check()

    def _check(self):
        if not isinstance(self.alpha, float):
            raise TypeError("alpha must be a float")
        
        if not isinstance(self.mol_times, int):
            raise TypeError("mol_times must be an int")
        
        if not isinstance(self.active_method, str):
            raise TypeError("method must be a string")
        
        if self.logic_domain is not None and not isinstance(self.logic_domain, TensorLike):
            raise TypeError("logic_domain must be tensor-like data or None")
        
        if self.tol is not None and not isinstance(self.tol, (int, float)):
            raise TypeError("tol must be an int, float or None")
        
        if not isinstance(self.maxit, int):
            raise TypeError("maxit must be an int")
        
        if not isinstance(self.is_pre, bool):
            raise TypeError("is_pre must be a bool")
        
        if not isinstance(self.pre_steps, int):
            raise TypeError("pre_steps must be an int")
        
        if self.fun_solver is not None and not callable(self.fun_solver):
            raise TypeError("fun_solver must be a callable or None")
        
        if not isinstance(self.monitor, str):
            raise TypeError("monitor must be a string")
        
        if not isinstance(self.mol_meth, str):
            raise TypeError("mol_meth must be a string")
        
        if not isinstance(self.int_meth, str):
            raise TypeError("int_meth must be a string")
        
        if self.int_meth not in ['comass'] and self.pde is None:
            raise ValueError("pde must be given when int_meth is not 'comass'")
        
        if self.parallel_mode not in ['none', 'thread', 'process']:
            raise ValueError("parallel_mode must be 'none', 'thread', 'process'")
