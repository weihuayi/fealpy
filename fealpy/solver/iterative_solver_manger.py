import threading
from typing import Optional, Dict, Any, List
import numpy as np
from ..backend import backend_manager as bm
from ..sparse import COOTensor, CSRTensor

class IterativeSolverNotAvailableError(Exception):
    """Exception indicating no iterative solver is available"""
    pass

class IterativeSolverManager:
    """
    Iterative solver manager: unified interface for iterative solvers with preconditioning.

    Usage:
        ism = IterativeSolverManager()
        ism.set_matrix(A, matrix_type='SP')
        ism.set_solver('cg')            # optional: if omitted, auto-select
        ism.set_pc('ilu')               # optional: if omitted, no preconditioner
        ism.set_tolerances(rtol=1e-5, atol=1e-5, maxit=5000)
        x = ism.solve(b)
    """
    _SOLVER_MAPPING: Dict[str, Any] = {}
    _PC_MAPPING: Dict[str, Any] = {}
    _cache = threading.local()

    @classmethod
    def register_solver(cls, name: Optional[str] = None):
        def decorator(sol_cls):
            key = name or sol_cls.__name__.lower()
            if key in cls._SOLVER_MAPPING:
                raise KeyError(f"Solver '{key}' already registered")
            cls._SOLVER_MAPPING[key] = sol_cls
            return sol_cls
        return decorator

    @classmethod
    def register_pc(cls, name: Optional[str] = None):
        def decorator(pc_cls):
            key = name or pc_cls.__name__.lower()
            if key in cls._PC_MAPPING:
                raise KeyError(f"Preconditioner '{key}' already registered")
            cls._PC_MAPPING[key] = pc_cls
            return pc_cls
        return decorator

    def __init__(self):
        self.A = None
        self.matrix_type: str = 'G'
        self.solver_name: Optional[str] = None
        self.pc_name: Optional[str] = None
        self.raw_kwargs: Dict[str, Any] = {}
        self.rtol: float = 1e-8
        self.atol: float = 1e-8
        self.maxit: int = 1000
        self._available_solvers: List[str] = []
        self._available_pcs: List[str] = []

    def set_matrix(self, A: Any, matrix_type: str = 'G') -> None:
        self.A = A
        self.matrix_type = matrix_type
        # clear cache
        for attr in ('solver', 'pc'):
            if hasattr(self._cache, attr):
                delattr(self._cache, attr)
        # update available lists
        self._available_solvers = list(self._SOLVER_MAPPING.keys())
        self._available_pcs = list(self._PC_MAPPING.keys())

    def _auto_select_solver(self) -> str:
        # SPD 优先 CG，通用优先 GMRES
        if self.matrix_type == 'SP':
            priority = ['cg', 'gmres', 'bicgstab', 'minres']
        else:
            priority = ['gmres', 'bicgstab', 'cg', 'minres']
        for name in priority:
            if name in self._available_solvers:
                return name
        raise IterativeSolverNotAvailableError(f"No available iterative solver: {self._available_solvers}")

    def set_solver(self, solver_name: Optional[str] = None, **kwargs) -> None:
        self.solver_name = solver_name or 'auto'
        if self.solver_name == 'auto':
            self.solver_name = self._auto_select_solver()
        if self.solver_name not in self._SOLVER_MAPPING:
            raise IterativeSolverNotAvailableError(f"Solver '{self.solver_name}' not registered; available: {self._available_solvers}")
        self.raw_kwargs = kwargs
        # clear solver cache
        if hasattr(self._cache, 'solver'):
            delattr(self._cache, 'solver')

    def set_pc(self, pc_name: Optional[str] = None, **kwargs) -> None:
        self.pc_name = pc_name
        if self.pc_name and self.pc_name not in self._PC_MAPPING:
            raise IterativeSolverNotAvailableError(f"Preconditioner '{self.pc_name}' not registered; available: {self._available_pcs}")
        self.raw_kwargs.update(kwargs)
        # clear pc cache
        if hasattr(self._cache, 'pc'):
            delattr(self._cache, 'pc')

    def set_tolerances(self, rtol: float = 1e-8, atol: float = 1e-8, maxit: int = 1000) -> None:
        self.rtol = rtol
        self.atol = atol
        self.maxit = maxit

    def get_preconditioner(self):
        if not self.pc_name:
            return None

        if not hasattr(self._cache, 'pc'):
            pc_cls = self._PC_MAPPING[self.pc_name]
            pc = pc_cls()

            if hasattr(pc, 'setup'):
                pc.setup(self.A, matrix_type=self.matrix_type)
                self._cache.pc = pc
            elif hasattr(pc, 'apply'):
                self._cache.pc = pc.apply(self.A)
            else:
                raise TypeError(f"Preconditioner '{self.pc_name}' must have either 'setup' or 'apply' method.")

        return self._cache.pc

    def get_solver(self):
        if not hasattr(self._cache, 'solver'):
            # wrap existing cg function or other registered classes
            SolverCls = self._SOLVER_MAPPING[self.solver_name]
            solver = SolverCls() if isinstance(SolverCls, type) else SolverCls
            # For function-based solvers like cg, wrap into callable object
            if callable(solver) and not hasattr(solver, 'setup'):
                # Create a simple wrapper
                class FuncSolver:
                    def __init__(self, func):
                        self.func = func
                    def setup(self, A, pc, rtol, atol, maxit):
                        self.A = A
                        self.pc = pc
                        self.rtol = rtol
                        self.atol = atol
                        self.maxit = maxit
                        
                    def solve(self, A, b):
                        M = (lambda x: self.pc.apply(x)) if self.pc else None
                        args = dict(atol=self.atol, rtol=self.rtol, maxit=self.maxit, M=M, returninfo=True)
                        x, info = self.func(A, b, **args)
                        return x, info
                    
                solver = FuncSolver(solver)
            solver.setup(self.A, pc=self.get_preconditioner(), rtol=self.rtol,
                         atol=self.atol, maxit=self.maxit, matrix_type=self.matrix_type)
            self._cache.solver = solver
            
        return self._cache.solver

    def solve(self, b: Any):
        if self.A is None:
            raise ValueError("Matrix not set. Call set_matrix() first.")
        if self.solver_name is None or self.solver_name == 'auto':
            self.solver_name = self._auto_select_solver()
        solver = self.get_solver()
        x = solver.solve(self.A, b)
       
        return x

@IterativeSolverManager.register_solver('cg')
class CGSolver:
    """
    Conjugate Gradient solver for symmetric positive-definite matrices.
    """
    def setup(self, A: COOTensor, pc=None, rtol=1e-8, atol=1e-8, maxit=5000, matrix_type='G'):
        self.A = A
        self.pc = pc
        self.rtol = rtol
        self.atol = atol
        self.maxit = maxit
        self.matrix_type = matrix_type

    def solve(self, A: COOTensor, b: np.ndarray) -> np.ndarray:
        from fealpy.solver import cg  
        x = cg(A, b, M=self.pc, rtol=self.rtol, atol=self.atol, maxit=self.maxit)
        return x

@IterativeSolverManager.register_solver('gmres')
class GMRESSolver:
    """
    Generalized Minimal RESidual (GMRES) solver for general non-symmetric linear systems.
    """
    def setup(self, A: COOTensor, pc=None, rtol=1e-8, atol=1e-8, maxit=5000, matrix_type='G'):
        self.A = A
        self.pc = pc
        self.rtol = rtol
        self.atol = atol
        self.maxit = maxit
        self.matrix_type = matrix_type

    def solve(self, A: COOTensor, b: np.ndarray) -> np.ndarray:
        from fealpy.solver import gmres  
        x = gmres(A, b, M=self.pc, rtol=self.rtol, atol=self.atol, maxit=self.maxit, restart=30)
        return x[0]
    
@IterativeSolverManager.register_solver('minres')
class MINRESSolver:
    """
    MINimum RESidual (MINRES) solver for symmetric (possibly indefinite) linear systems.
    """
    def setup(self, A: COOTensor, pc=None, rtol=1e-8, atol=1e-8, maxit=5000, matrix_type='G'):
        self.A = A
        self.pc = pc
        self.rtol = rtol
        self.atol = atol
        self.maxit = maxit
        self.matrix_type = matrix_type

    def solve(self, A: COOTensor, b: np.ndarray) -> np.ndarray:
        from fealpy.solver import minres  
        x = minres(A, b, M=self.pc, rtol=self.rtol, atol=self.atol, maxit=self.maxit)
        return x[0]

@IterativeSolverManager.register_solver('lgmres')
class LGMRESSolver:
    """
    Loose Generalized Minimal Residual (LGMRES) solver for general non-symmetric linear systems.
    """
    def setup(self, A: COOTensor, pc=None, rtol=1e-8, atol=1e-8, maxit=5000, matrix_type='G'):
        self.A = A
        self.pc = pc
        self.rtol = rtol
        self.atol = atol
        self.maxit = maxit
        self.matrix_type = matrix_type

    def solve(self, A: COOTensor, b: np.ndarray) -> np.ndarray:
        from fealpy.solver import lgmres  
        x = lgmres(A, b, M=self.pc, rtol=self.rtol, atol=self.atol, maxit=self.maxit, inner_m=20, outer_k=3)
        return x[0]

@IterativeSolverManager.register_solver('bicg')
class BiCGSolver:
    """
    BiConjugate Gradient (BiCG) solver for general non-symmetric linear systems.
    """
    def setup(self, A: COOTensor, pc=None, rtol=1e-8, atol=1e-8, maxit=5000, matrix_type='G'):
        self.A = A
        self.pc = pc
        self.rtol = rtol
        self.atol = atol
        self.maxit = maxit
        self.matrix_type = matrix_type

    def solve(self, A: COOTensor, b: np.ndarray) -> np.ndarray:
        from fealpy.solver import bicg 
        x = bicg(A, b, M=self.pc, rtol=self.rtol, atol=self.atol, maxit=self.maxit)
        return x[0]
    
@IterativeSolverManager.register_solver('bicgstab')
class BiCGSTABSolver:
    """
    BiConjugate Gradient Stabilized (BiCGSTAB) solver for general non-symmetric linear systems."
    """
    def setup(self, A: COOTensor, pc=None, rtol=1e-8, atol=1e-8, maxit=5000, matrix_type='G'):
        self.A = A
        self.pc = pc
        self.rtol = rtol
        self.atol = atol
        self.maxit = maxit
        self.matrix_type = matrix_type

    def solve(self, A: COOTensor, b: np.ndarray) -> np.ndarray:
        from fealpy.solver import bicgstab 
        x = bicgstab(A, b, M=self.pc, rtol=self.rtol, atol=self.atol, maxit=self.maxit)
        return x[0]
    
@IterativeSolverManager.register_pc('jacobi')
class JacobiPreconditioner:
    """
    Jacobi preconditioner for iterative solvers.
    """
    def apply(self,A):
        diags = A.diags()
        return CSRTensor(diags.crow, diags.col, 1/diags.values, A.shape)

@IterativeSolverManager.register_pc('mg')   
class GAMGPreconditioner:
    
    def apply(self, A):
        from fealpy.solver import GAMGSolver
        solver = GAMGSolver()
        solver.setup(A)
        pre_matrix = solver.preconditioner()
        return pre_matrix