import threading
from .. import logger
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import asyncio
import inspect
from ..backend import backend_manager as bm
from ..sparse import COOTensor, CSRTensor
import numpy as np

class SolverNotAvailableError(Exception):
    """Custom exception indicating no solver is available"""
    pass

class DirectSolverManager:
    """
    Direct solver manager: provides a unified interface for various sparse linear system solvers.

    Usage example:
        mgr = DirectSolverManager(solver_name='auto')
        mgr.set_matrix(A, matrix_type='U')
        x = mgr.solve(b)
    """
    _SOLVER_MAPPING: Dict[str, type] = {}
    _solver_cache = threading.local()

    @classmethod
    def register(cls, name: Optional[str] = None):
        """
        Decorator to register a new solver class.
        Parameters:
            name: the key under which to register the solver; defaults to the class name in lowercase.
        """
        def decorator(solver_cls):
            key = name or solver_cls.__name__.lower()
            if key in cls._SOLVER_MAPPING:
                raise KeyError(f"Solver '{key}' already registered")
            cls._SOLVER_MAPPING[key] = solver_cls
            return solver_cls
        return decorator

    def __init__(self, solver_name: str = 'auto'):
        """
        Initialize the DirectSolverManager.
        Parameters:
            solver_name: the name of the solver to use or 'auto' to select automatically.
        """
        self.solver_name = solver_name
        self.A = None
        self.device = None
        self._available: List[str] = []
        self._matrix_type: str = 'G'
        self.raw_kwargs: Dict[str, Any] = {}

    @classmethod
    def available_solvers(cls) -> List[str]:
        """Return the list of all registered solver names."""
        return list(cls._SOLVER_MAPPING.keys())

    def set_matrix(self, A: Any, matrix_type: str = 'G') -> None:
        """
        Set the system matrix and its type; clears any cached solver and updates available solvers.

        Parameters:
            A: COOTensor or CSRTensor representing the system matrix.
            matrix_type: 'G' for general, 'L' for lower triangular, 'U' for upper triangular, 'SP' for symmetric positive definite.
        """
        self.A = A
        self._matrix_type = matrix_type
        self.device = A.indices.device
        
        # Clear thread-local solver cache
        for attr in ('solver', 'solver_name'):
            if hasattr(self._solver_cache, attr):
                delattr(self._solver_cache, attr)

        # Update list of available solvers
        self._available = self._filter_solvers()

    def _filter_solvers(self) -> List[str]:
        """
        Filter registered solvers by device type: use cupy on GPU, others on CPU.
        """
        if isinstance(self.device, str) and self.device.startswith('cuda'):
            candidates = ['cupy']
        else:
            candidates = ['scipy', 'mumps', 'pardiso', 'cholmod', 'cupy']
        return [s for s in candidates if s in self._SOLVER_MAPPING]

    def _auto_select(self) -> str:
        """
        Automatically select a solver from available ones by priority.
        Priority order: scipy, pardiso, mumps, cholmod, cupy.
        """
        for name in ['scipy', 'pardiso', 'mumps', 'cholmod', 'cupy']:
            if name in self._available:
                return name
        raise SolverNotAvailableError(
            f"No available solver: device={self.device}, candidates={self._available}"
        )

    def set_solver(self, solver_name: Optional[str] = None, **kwargs) -> None:
        """
        Explicitly set the solver and its initialization parameters; matrix_type is ignored here.

        Parameters:
            solver_name: name of the solver to use or 'auto' to select automatically.
            kwargs: parameters passed to the solver constructor (excluding matrix_type).
        """
        # Filter out any matrix_type passed by mistake
        self.raw_kwargs = {k: v for k, v in kwargs.items() if k != 'matrix_type'}
        self.solver_name = solver_name if solver_name is not None else self.solver_name
        if self.solver_name in ('auto'):
            self.solver_name = self._auto_select()

        if self.solver_name not in self._SOLVER_MAPPING:
            raise SolverNotAvailableError(f"Solver '{self.solver_name}' not available; candidates: {self._available}")

        # Clear old instance if switching solver
        old = getattr(self._solver_cache, 'solver_name', None)
        if old != self.solver_name and hasattr(self._solver_cache, 'solver'):
            delattr(self._solver_cache, 'solver')

        # Use cached solver if unchanged
        if hasattr(self._solver_cache, 'solver') and old == self.solver_name:
            logger.debug(f"Using cached solver '{self.solver_name}'")
            return

        # Instantiate new solver
        solver_cls = self._SOLVER_MAPPING[self.solver_name]
        sig = inspect.signature(solver_cls.__init__)
        valid = {p for p in sig.parameters if p not in ('self', 'kwargs')}
        init_kwargs = {k: v for k, v in self.raw_kwargs.items() if k in valid}
        unused = set(self.raw_kwargs) - valid
        if unused:
            logger.warning(f"Parameters {unused} ignored by solver '{self.solver_name}'")
        solver = solver_cls(**init_kwargs)
        self._solver_cache.solver = solver
        self._solver_cache.solver_name = self.solver_name

    def get_solver(self) -> Any:
        """Get the current solver instance, instantiating it if needed."""
        if not hasattr(self._solver_cache, 'solver'):
            self.set_solver(self.solver_name)
        return self._solver_cache.solver

    def _clear_cache(self) -> None:
        """Clear the thread-local solver cache."""
        for attr in ('solver', 'solver_name'):
            if hasattr(self._solver_cache, attr):
                delattr(self._solver_cache, attr)

    def solve(self, b: Any):
        """
        Synchronously solve A x = b, then clear the solver cache.

        Parameters:
            b: right-hand side vector or tensor.
        Returns:
            x: solution as a NumPy ndarray.
        """
        if self.A is None:
            raise ValueError("Matrix not set, call set_matrix() first.")
        solver = self.get_solver()
        x = solver.solve(self.A, b, matrix_type=self._matrix_type)
        self._clear_cache()
        return x

    async def solve_async(self, b: Any) -> np.ndarray:
        """
        Asynchronously solve A x = b, then clear the solver cache.

        Parameters:
            b: right-hand side vector or tensor.
        Returns:
            x: solution as a NumPy ndarray.
        """
        if self.A is None:
            raise ValueError("Matrix not set, call set_matrix() first.")
        solver = self.get_solver()
        arr = bm.to_numpy(b) if not isinstance(b, np.ndarray) else b
        if hasattr(solver, 'solve_async') and inspect.iscoroutinefunction(solver.solve_async):
            x = await solver.solve_async(self.A, arr, matrix_type=self._matrix_type)
        else:
            loop = asyncio.get_event_loop()
            x = await loop.run_in_executor(
                None,
                lambda: solver.solve(self.A, arr, matrix_type=self._matrix_type)
            )
        self._clear_cache()
        return x

class BaseSolver(ABC):
    """Abstract base class for all solver implementations."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def solve(self, A: Any, b: np.ndarray, matrix_type: str = 'G') :
        """Solve the linear system A x = b."""
        pass

@DirectSolverManager.register('scipy')
class ScipySolver(BaseSolver):
    """Solver using scipy.sparse.linalg."""

    def solve(self, A, b, matrix_type: str = 'G') :
        if isinstance(A, COOTensor):
            A = A.to_scipy().tocsr()
        elif isinstance(A, CSRTensor):
            A = A.to_scipy()
        b = bm.to_numpy(b) if not isinstance(b, np.ndarray) else b
        from scipy.sparse.linalg import spsolve, spsolve_triangular
        if matrix_type == 'U':
            return bm.tensor(spsolve_triangular(A, b, lower=False, **self.kwargs))
        if matrix_type == 'L':
            return bm.tensor(spsolve_triangular(A, b, lower=True, **self.kwargs))
        else:
            return bm.tensor(spsolve(A, b, **self.kwargs))

@DirectSolverManager.register('mumps')
class MumpsSolver(BaseSolver):
    """Solver using MUMPS (via pymumps)."""
    

    def solve(self, A, b, matrix_type: str = 'G') :
        if isinstance(A, COOTensor):
            A = A.to_scipy().tocsr()
        elif isinstance(A, CSRTensor):
            A = A.to_scipy()
        b = bm.to_numpy(b) if not isinstance(b, np.ndarray) else b
        from mumps import DMumpsContext
        sym_map = {'G': 0, 'SP': 1, 'S': 2}
        ctx = DMumpsContext(par=1, sym=sym_map.get(matrix_type, 0), comm=None)
        ctx.set_centralized_sparse(A)
        x = b.copy()
        ctx.set_rhs(x)
        ctx.set_silent()
        ctx.run(job=6)
        ctx.destroy()
        return bm.tensor(x)

@DirectSolverManager.register('pardiso')
class PardisoSolver(BaseSolver):
    """Solver using pypardiso."""

    def solve(self, A, b: np.ndarray, matrix_type: str = 'G') :
        if isinstance(A, COOTensor):
            A = A.to_scipy().tocsr()
        elif isinstance(A, CSRTensor):
            A = A.to_scipy()
        b = bm.to_numpy(b) if not isinstance(b, np.ndarray) else b
        from pypardiso import PyPardisoSolver
        mapping = {'G': 11, 'SP': 1, 'S': 1}
        solver = PyPardisoSolver()
        solver.set_matrix_type(mapping.get(matrix_type.upper(), 11))
        solver.factorize(A)
        x = solver.solve(A, b)
        solver.free_memory(everything=True)
        return bm.tensor(x)

@DirectSolverManager.register('cholmod')
class CholmodSolver(BaseSolver):
    """Solver using scikit-sparse Cholmod."""

    def solve(self, A, b, matrix_type: str = 'G') -> np.ndarray:
        if matrix_type != 'SP':
            raise ValueError("CholmodSolver only supports SP type")
        if isinstance(A, COOTensor):
            A = A.to_scipy().tocsr()
        elif isinstance(A, CSRTensor):
            A = A.to_scipy()
        b = bm.to_numpy(b) if not isinstance(b, np.ndarray) else b
        from sksparse.cholmod import cholesky
        factor = cholesky(A)
        return bm.tensor(factor(b))

@DirectSolverManager.register('cupy')
class CupySolver(BaseSolver):
    """Solver using CuPy for GPU-based sparse solutions."""
    
    def _to_cupy_data(self, A, b):
        import cupy as cp
        if isinstance(A.indices, np.ndarray): # numpy backend
            A =  A.to_scipy() 
            A = cp.sparse.csr_matrix(A.astype(cp.float64))
        elif A.indices.device.type == "cpu": # torch backend
            A = A.device_put("cuda")
            indices = cp.from_dlpack(A.indices)
            data = cp.from_dlpack(A.values)
            A = cp.sparse.csr_matrix((data, (indices[0], indices[1])), shape=A.shape)
        else:
            indices = cp.from_dlpack(A.indices)
            data = cp.from_dlpack(A.values)
            A = cp.sparse.csr_matrix((data, (indices[0], indices[1])), shape=A.shape)

        if isinstance(b, np.ndarray) or b.device.type == "cpu":
            b = bm.to_numpy(b)
            b = cp.array(b)
        else:
            b = cp.from_dlpack(b)
        return A, b

    def solve(self, A, b, matrix_type: str = 'G') -> np.ndarray:
        import cupy as cp
        A_gpu, b_gpu = self._to_cupy_data(A, b)
        if matrix_type == 'U':
            from cupyx.scipy.sparse.linalg import spsolve_triangular
            x = spsolve_triangular(A_gpu, b_gpu, lower=False, **self.kwargs)
        elif matrix_type == 'L':
            from cupyx.scipy.sparse.linalg import spsolve_triangular
            x = spsolve_triangular(A_gpu, b_gpu, lower=True, **self.kwargs)
        else:
            from cupyx.scipy.sparse.linalg import spsolve
            x = spsolve(A_gpu, b_gpu, **self.kwargs)

        return bm.tensor(x)
