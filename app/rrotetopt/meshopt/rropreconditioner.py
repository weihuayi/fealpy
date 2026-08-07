from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.solver import cg
from fealpy.operator import LinearOperator
from fealpy.backend import TensorLike

from opt import BasePreconditioner

class ProjectedCGPreconditioner(BasePreconditioner):
    def __init__(
        self,
        problem,
        update_interval: int = 3,
        rtol: float = 1e-2,
        maxiter: int = 100,
    ) -> None:
        super().__init__(update_interval=update_interval)
        self.problem = problem
        self.rtol = rtol
        self.maxiter = maxiter

        self.P = None
        self.Pi = None
        self._Pproj = None
        self._cg_x0 = None

    def setup(self, x: TensorLike) -> None:
        self.P, self.Pi = self.problem.build_preconditioner_matrices(x)
        self._build_projected_operator()

    def update(self, x: TensorLike, k: Optional[int] = None) -> None:
        if k is None or (self.update_interval > 0 and k % self.update_interval == 0):
            self.P, self.Pi = self.problem.build_preconditioner_matrices(x)
            self._build_projected_operator()

    def _build_projected_operator(self) -> None:
        P = self.P
        Pi = self.Pi
        n = Pi.shape[1]

        def mv(v):
            return Pi.T @ (P @ (Pi @ v))

        self._Pproj = LinearOperator((n, n), matvec=mv)
        self._cg_x0 = None

    def apply(self, q: TensorLike) -> TensorLike:
        """
        返回 (Pi^T P Pi)^{-1} (Pi^T q) 的近似解
        """
        if self.P is None or self.Pi is None or self._Pproj is None:
            return q

        rhs = self.Pi @ q
        z, info = cg(
            self._Pproj,
            rhs,
            x0=self._cg_x0,
            rtol=self.rtol,
            #maxiter=self.maxiter,
            maxit=self.maxiter,
            returninfo = True
        )
        res = info['residual']
        if res< self.rtol*bm.linalg.norm(rhs):
            self._cg_x0 = z
            return z
        # 失败时回退到不预条件
        return q

    def scale(self, s: TensorLike, y: TensorLike) -> float:
        """
        gamma = (s^T y) / (y^T M^{-1} y)
        """
        if self.P is None or self.Pi is None or self._Pproj is None:
            return super().scale(s, y)

        rhs = self.Pi @ y
        z, info = cg(
            self._Pproj,
            rhs,
            rtol=self.rtol,
            maxit=self.maxiter,
            returninfo = True
        )
        res = info['residual']
        if res< self.rtol*bm.linalg.norm(rhs):
            return super().scale(s, y)
        den = float(y @ z)
        sty = float(s @ y)
        if den > 0:
            return sty / den
        return 1.0

    def reset(self) -> None:
        self.P = None
        self.Pi = None
        self._Pproj = None
        self._cg_x0 = None
