
from .nodetype import CNodeType, PortConf, DataType

__all__ = ["CGSolver",
           "DirectSolver",
           "IterativeSolver",
           "ScipyEigenSolver",
           "SLEPcEigenSolver",
           "MGStokesSolver"
           ]


class CGSolver(CNodeType):
    TITLE: str = "CG 解法器"
    PATH: str = "simulation.solvers"
    INPUT_SLOTS = [
        PortConf("A", DataType.LINOPS, title="算子"),
        PortConf("b", DataType.TENSOR, title="向量"),
        PortConf("x0", DataType.TENSOR, title="初值", default=None),
        PortConf("maxit", DataType.INT, title="最大迭代数", default=10000, min_val=1)
    ]
    OUTPUT_SLOTS = [
        PortConf("out", DataType.TENSOR, title="结果")
    ]

    @staticmethod
    def run(*args, **kwargs):
        from ..solver import cg
        return cg(*args, **kwargs)


class DirectSolver(CNodeType):
    TITLE: str = "直接解法器"
    PATH: str = "simulation.solvers"

    INPUT_SLOTS = [
        PortConf("A", DataType.LINOPS, title="矩阵"),
        PortConf("b", DataType.TENSOR, title="向量"),
        PortConf("matrix_type", DataType.MENU, title="矩阵类型",
                 default="G", items=["U", "L", "S", "SP", "G"]),
        PortConf("solver", DataType.MENU, title="求解器",
                 default="scipy", items=["scipy", "mumps", "pardiso", "cupy"])
    ]
    OUTPUT_SLOTS = [
        PortConf("out", DataType.TENSOR, title="结果")
    ]

    @staticmethod
    def run(*args, **kwargs):

        A = kwargs.get("A")
        b = kwargs.get("b")
        matrix_type = kwargs.get("matrix_type", kwargs.get("matrixType"))
        solver = kwargs.get("solver")
        from ..solver import DirectSolverManager
        mgr = DirectSolverManager()
        mgr.set_solver(solver_name=solver)
        mgr.set_matrix(A, matrix_type=matrix_type)
        x = mgr.solve(b)

        return x
    

class IterativeSolver(CNodeType):
    TITLE: str = "迭代解法器"
    PATH: str = "simulation.solvers"
    INPUT_SLOTS = [
        PortConf("A", DataType.LINOPS, title="算子"),
        PortConf("b", DataType.TENSOR, title="向量"),
        PortConf("maxit", DataType.INT, title="最大迭代数", default=10000),
        PortConf("rtol", DataType.FLOAT, title="相对容忍度", default=1e-4),
        PortConf("atol", DataType.FLOAT, title="绝对容忍度", default=1e-4),
        PortConf("solver", DataType.MENU, title="求解器",
                 default="gmres", items=["gmres", "minres", "cg", "lgmres",  "bicg"])
    ]
    OUTPUT_SLOTS = [
        PortConf("out", DataType.TENSOR, title="结果")
    ]
    @staticmethod
    def run(*args, **kwargs):
        A = kwargs.get("A")
        b = kwargs.get("b")

        maxit = kwargs.get("maxit")
        rtol = kwargs.get("rtol")
        atol = kwargs.get("atol")
        solver = kwargs.get("solver")
        from ..solver import IterativeSolverManager
        mgr = IterativeSolverManager()
        mgr.set_solver(solver_name=solver)
        mgr.set_matrix(A)
        mgr.set_tolerances(rtol=rtol, atol=atol, maxit=maxit)
        x = mgr.solve(b)
        return x


class ScipyEigenSolver(CNodeType):
    TITLE: str = "Scipy特征值求解器"
    PATH: str = "simulation.solvers"
    INPUT_SLOTS = [
        PortConf("S", DataType.TENSOR, title="刚度矩阵 S"),
        PortConf("M", DataType.TENSOR, title="质量矩阵 M"),
        PortConf("neigen", DataType.INT, title="求取特征值个数", default=6, min_val=1),
        PortConf("which", DataType.STRING, title="求解特征值类型", default='SM'),
    ]
    OUTPUT_SLOTS = [
        PortConf("val", DataType.TENSOR, title="特征值"),
        PortConf("vec", DataType.TENSOR, title="特征向量"),
    ]
    
    @staticmethod    
    def run(*args, **kwargs):
        from scipy.sparse.linalg import eigsh
        S = kwargs.get('S')
        M = kwargs.get('M')
        neigen = kwargs.get('neigen')
        val, vec = eigsh(S, k=neigen, M=M, which=kwargs.get('which', 'SM'), tol=1e-6, maxiter=1000)
        vec = vec.T
        return val, vec
    

class SLEPcEigenSolver(CNodeType):
    TITLE: str = "SLEPc特征值求解器"
    PATH: str = "simulation.solvers"
    INPUT_SLOTS = [
        PortConf("S", DataType.TENSOR, title="刚度矩阵 S"),
        PortConf("M", DataType.TENSOR, title="质量矩阵 M"),
        PortConf("neigen", DataType.INT, title="求取特征值个数", default=6, min_val=1),
        PortConf("sigma", DataType.FLOAT, title="目标值", default=0.0),
    ]
    OUTPUT_SLOTS = [
        PortConf("val", DataType.TENSOR, title="特征值"),
        PortConf("vec", DataType.TENSOR, title="特征向量"),
    ]
    
    @staticmethod    
    def run(S, M, neigen, sigma):
        
        from petsc4py import PETSc
        from slepc4py import SLEPc
        from ..backend import backend_manager as bm
        PS = PETSc.Mat().createAIJ(
                size=S.shape, 
                csr=(S.indptr, S.indices, S.data))
        PS.assemble()
        PM = PETSc.Mat().createAIJ(
                size=M.shape, 
                csr=(M.indptr, M.indices, M.data))
        PS.assemble()
        

        eps = SLEPc.EPS().create()
        eps.setOperators(PS, PM)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        #eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        eps.setType(SLEPc.EPS.Type.LANCZOS)

        opts = PETSc.Options()
        opts['eps_lanczos_reorthog'] = 'local'
        eps.setFromOptions()

        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
        eps.setTarget(sigma)  # ← 显式设置目标

        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(sigma)

        ksp = st.getKSP()
        ksp.setType('preonly')
        pc = ksp.getPC()
        pc.setType('lu')
        try:
            pc.setFactorSolverType('mumps')
        except Exception:
            pc.setFactorSolverType('superlu')
        vec = PS.getVecRight()
        vec.setRandom()
        eps.setInitialSpace([vec])
        eps.setDimensions(nev=neigen, ncv=min(8*neigen, 200))
        eps.setTolerances(tol=1e-6, max_it=10000)
        
        nn = S.shape[0]
        v0 = bm.random.rand(nn) + 10000
        v0 = PETSc.Vec().createWithArray(v0)
        eps.setInitialSpace([v0])       
        
        eps.solve()
        
        eigvals = []
        eigvecs = []

        vr, vi = eps.getOperators()[0].getVecs()
        print(f"Number of eigenvalues converged: {eps.getConverged()}")
        for i in range(min(neigen, eps.getConverged())):
            val = eps.getEigenpair(i, vr, vi)
            eigvals.append(val.real)
            eigvecs.append(vr.getArray().copy())
        val = bm.array(eigvals)
        vec = bm.array(eigvecs)


        return val, vec
    

class MGStokesSolver(CNodeType):
    TITLE: str = "Stokes多重网格求解器"
    PATH: str = "simulation.solvers"
    
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="三棱柱网格"),
        PortConf("op", DataType.LINOPS, title="初始系数矩阵"),
        PortConf("idx", DataType.TENSOR, title="三棱柱网格节点映射"),
        PortConf("F1", DataType.TENSOR, title="右端向量"),
        PortConf("bd_flag", DataType.TENSOR, title="边界条件"),
        PortConf("ugdof", DataType.INT, title="初始速度自由度"),
        PortConf("Ai", DataType.LINOPS, title="每层的动量方程刚度矩阵"),
        PortConf("Bi", DataType.LINOPS, title="每层的散度矩阵"),
        PortConf("Bti", DataType.LINOPS, title="每层的梯度矩阵"),
        PortConf("bigAi", DataType.TENSOR, title="最粗网格的整体矩阵"),
        PortConf("P_u", DataType.LINOPS, title="速度插值矩阵"),
        PortConf("R_u", DataType.LINOPS, title="速度限制矩阵"),
        PortConf("P_p", DataType.LINOPS, title="压力插值矩阵"),
        PortConf("R_p", DataType.LINOPS, title="压力限制矩阵"),
        PortConf("Nu", DataType.INT, title="每层速度自由度"),
        PortConf("Np", DataType.INT, title="每层压力自由度"),
        PortConf("level", DataType.INT, title="多重网格层数"),
        PortConf("auxMat", DataType.MENU, title="每层LSC-DGS平滑所需若干参数"),
        PortConf("options", DataType.MENU, title="多重网格所需若干参数"),
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="三棱柱网格"),
        PortConf("uh", DataType.TENSOR, title="速度解"),
        PortConf("ph", DataType.TENSOR, title="压力解"),
    ]
    
    @staticmethod    
    def run(**kwargs):
        from ..backend import bm
        from ..solver import MGStokes
        mesh = kwargs.get('mesh')
        op = kwargs.get('op')
        idx = kwargs.get('idx')
        F1 = kwargs.get('F1')
        bd_flag = kwargs.get('bd_flag')
        ugdof = kwargs.get('ugdof')
        Ai = kwargs.get('Ai')
        Bi = kwargs.get('Bi')
        Bti = kwargs.get('Bti')
        bigAi = kwargs.get('bigAi')
        P_u = kwargs.get('P_u')
        R_u = kwargs.get('R_u')
        P_p = kwargs.get('P_p')
        R_p = kwargs.get('R_p')
        Nu = kwargs.get('Nu')
        Np = kwargs.get('Np')
        level = kwargs.get('level')
        auxMat = kwargs.get('auxMat')
        options = kwargs.get('options')

        Solver = MGStokes(Ai, Bi, Bti, bigAi,
                P_u, R_u, P_p, R_p,
                Nu, Np, level, 
                auxMat, options)
        x_in = Solver.solve(op, F1[~bd_flag])
        x = bm.set_at(F1, ~bd_flag, x_in)
        uh = x[:3*ugdof].reshape(3,-1).T[idx,:]
        ph = x[3*ugdof:]
        print(x.max(), x.min())
        return mesh, uh, ph