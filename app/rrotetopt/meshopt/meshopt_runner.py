from dataclasses import dataclass
from typing import Optional, Callable, Literal, Dict, Any

from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike 

from .TriMeshProblem import TriMeshProblem
from .TriSurfMeshProblem import TriSurfMeshProblem
from .TetMeshProblem import TetMeshProblem
from .TriAniMeshProblem import TriAniMeshProblem
from .TetAniMeshProblem import TetAniMeshProblem
from opt.PLBFGSAlg import PLBFGS
from opt.PNLCGAlg import PNLCG

import matplotlib.pyplot as plt

@dataclass
class ConstraintSpec:
    """
    几何约束与边界约束描述
    """
    Project: Optional[Callable[[TensorLike], TensorLike]] = None
    Tangent1d: Optional[Callable[[TensorLike, TensorLike], TensorLike]] = None
    Normal2d: Optional[Callable[[TensorLike], TensorLike]] = None
    Metric: Optional[Callable[[TensorLike], TensorLike]] = None

    isFixNode: Optional[TensorLike] = None
    isBdEdgeNode: Optional[TensorLike] = None
    isBdFaceNode: Optional[TensorLike] = None
    FixAllBoundary: bool = False


@dataclass
class MeshOptConfig:
    """
    优化配置
    """
    problem_type: Literal["tri", "trisurf", "triani","tet","tetani"] = "tet"
    optimizer: Literal["LBFGS", "NLCG"] = "LBFGS"

    use_preconditioner: bool = False
    precond_update_interval: int = 3
    precond_rtol: float = 1e-2
    precond_maxiter: int = 100

    step_length: float = 1.0
    fun_val_diff: float = 1e-6
    step_length_tol: float = 1e-6
    norm_grad_tol: float = 1e-6
    max_iters: int = 200
    num_grad: int = 10
    print_info: bool = False

    flip: bool = True
    max_outer_iters: int = 20

    vtk_name: Optional[str] = None


class MeshOptimizerRunner:
    def __init__(
        self,
        mesh,
        constraints: ConstraintSpec,
        config: MeshOptConfig
    ) -> None:
        self.mesh = mesh
        self.constraints = constraints
        self.config = config

    # -----------------------------------------------------------------
    # 构造 Problem
    # -----------------------------------------------------------------
    def build_problem(self):
        c = self.constraints
        cfg = self.config

        if cfg.problem_type == "tri":
            options = TriMeshProblem.get_options(
                mesh=self.mesh,
                FixAllBoundary=c.FixAllBoundary,
                isFixNode=c.isFixNode,
                isBdEdgeNode=c.isBdEdgeNode,
                Project=c.Project,
                Tangent1d=c.Tangent1d,
            )
            problem = TriMeshProblem(options)

        elif cfg.problem_type == "trisurf":
            options = TriSurfMeshProblem.get_options(
                mesh=self.mesh,
                isFixNode=c.isFixNode,
                isBdEdgeNode=c.isBdEdgeNode,
                isBdFaceNode=c.isBdFaceNode,
                Project=c.Project,
                Tangent1d=c.Tangent1d,
                Normal2d=c.Normal2d,
            )
            problem = TriSurfMeshProblem(options)

        elif cfg.problem_type == "triani":
            options = TriAniMeshProblem.get_options(
                mesh=self.mesh,
                FixAllBoundary=c.FixAllBoundary,
                isFixNode=c.isFixNode,
                isBdEdgeNode=c.isBdEdgeNode,
                Project=c.Project,
                Tangent1d=c.Tangent1d,
                Metric = c.Metric,
            )
            problem = TriAniMeshProblem(options)

        elif cfg.problem_type == "tet":
            options = TetMeshProblem.get_options(
                mesh=self.mesh,
                FixAllBoundary=c.FixAllBoundary,
                isFixNode=c.isFixNode,
                isBdEdgeNode=c.isBdEdgeNode,
                isBdFaceNode=c.isBdFaceNode,
                Project=c.Project,
                Tangent1d=c.Tangent1d,
                Normal2d=c.Normal2d,
            )
            problem = TetMeshProblem(options)

        elif cfg.problem_type == "tetani":
            options = TetAniMeshProblem.get_options(
                mesh=self.mesh,
                FixAllBoundary=c.FixAllBoundary,
                isFixNode=c.isFixNode,
                isBdEdgeNode=c.isBdEdgeNode,
                isBdFaceNode=c.isBdFaceFode,
                Project=c.Project,
                Tangent1d=c.Tangent1d,
                Normal2d=c.Normal2d,
                Metric = c.Metric,
            )
            problem = TetAniMeshProblem(options)
       
        else:
            raise ValueError(f"Unknown problem_type: {cfg.problem_type}")

        # 统一设置优化参数
        problem.StepLength = cfg.step_length
        problem.FunValDiff = cfg.fun_val_diff
        problem.StepLengthTol = cfg.step_length_tol
        problem.NormGradTol = cfg.norm_grad_tol
        problem.MaxIters = cfg.max_iters
        problem.NumGrad = cfg.num_grad
        problem.Print = cfg.print_info

        # 挂预条件子
        if cfg.use_preconditioner:
            if not hasattr(problem, "build_preconditioner"):
                raise AttributeError(
                    f"{type(problem).__name__} does not provide build_preconditioner()."
                )
            problem.Preconditioner = problem.build_preconditioner(
                update_interval=cfg.precond_update_interval,
                rtol=cfg.precond_rtol,
                maxiter=cfg.precond_maxiter,
            )
        else:
            problem.Preconditioner = None

        return problem

    # -----------------------------------------------------------------
    # 构造优化器
    # -----------------------------------------------------------------
    def build_optimizer(self, problem):
        if self.config.optimizer == "LBFGS":
            return PLBFGS(problem)
        elif self.config.optimizer == "NLCG":
            return PNLCG(problem)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    # -----------------------------------------------------------------
    # 将优化变量 x 写回 mesh.node
    # 这里兼容有固定点和无固定点两种情况
    # -----------------------------------------------------------------
    def write_back_solution(self, x: NDArray, problem) -> None:
        node = self.mesh.entity("node")
        gd = node.shape[1]

        if hasattr(problem, "isFixNode") and problem.isFixNode is not None:
            is_free_node = ~problem.isFixNode
        else:
            is_free_node = bm.ones(node.shape[0], dtype=bm.bool)

        nfree = bm.sum(is_free_node)

        if gd == 2:
            node[is_free_node, 0] = x[:nfree]
            node[is_free_node, 1] = x[nfree:2 * nfree]
        elif gd == 3:
            node[is_free_node, 0] = x[:nfree]
            node[is_free_node, 1] = x[nfree:2*nfree]
            node[is_free_node, 2] = x[2*nfree:]
        else:
            raise ValueError(f"Unsupported geometric dimension: {gd}")

    # -----------------------------------------------------------------
    # flip 后更新 problem.x0 / mesh / 预条件子
    # -----------------------------------------------------------------
    def update_problem_after_flip(self, problem) -> None:
        node = self.mesh.entity("node")

        if hasattr(problem, "isFixNode") and problem.isFixNode is not None:
            is_free_node = ~problem.isFixNode
            x0 = node[is_free_node].T.flatten()
        else:
            x0 = node.T.flatten()

        problem.mesh = self.mesh
        problem.x0 = x0

        if getattr(problem, "Preconditioner", None) is not None:
            problem.Preconditioner.reset()

    # -----------------------------------------------------------------
    # 统一执行入口
    # -----------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        problem = self.build_problem()
        outer_iter = 0

        x = None
        f = None
        g = None
        opt = self.build_optimizer(problem)

        while True:
            outer_iter += 1

            x, f, g = opt.run()

            self.write_back_solution(x, problem)
            problem.mesh = self.mesh

            if not self.config.flip:
                break

            if not hasattr(problem, "flipopt") or not callable(problem.flipopt):
                break

            changed = problem.flipopt()
            self.mesh = problem.mesh

            if not changed:
                break

            self.update_problem_after_flip(problem)

            if outer_iter >= self.config.max_outer_iters:
                print(f"Reached maximum outer iterations: {self.config.max_outer_iters}")
                break

        if self.config.vtk_name is not None:
            self.mesh.to_vtk(fname=self.config.vtk_name)

        return {
            "mesh": self.mesh,
            "x": x,
            "f": f,
            "g": g,
            "problem": problem,
            "optimizer": opt,
            "outer_iterations": outer_iter,
        }


def optimize_mesh(mesh, constraints: ConstraintSpec, config: MeshOptConfig):
    """
    函数式接口，方便用户直接调用
    """
    runner = MeshOptimizerRunner(mesh, constraints, config)
    return runner.run()

