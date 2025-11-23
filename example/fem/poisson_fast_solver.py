
from typing import Optional, TypeVar, Union, Generic, Callable
from fealpy.typing import TensorLike, Index, _S, Threshold

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh, TetrahedronMesh
from fealpy.decorator import barycentric, cartesian

from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import BilinearForm, ScalarDiffusionIntegrator, ScalarMassIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator
from fealpy.fem import DirichletBC
from fealpy.model import PDEModelManager
from fealpy.pde.poisson_2d import CosCosData
from fealpy.pde.poisson_3d import CosCosCosData

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator

import time
import sys
import sympy as sp

from scipy.sparse.linalg import cg
from pyamg.relaxation.relaxation import gauss_seidel
import pyamg

class ASPForPoisson(object):
    """
    Add Schwarz Preconditioner for Poisson equation.
    Space decomposition based on vertices:
                        V = V_0 + (V_1 + V_2 + ... + V_N)
    V_0 is the linear finite element space. V_i is the subspace associated with
    vertex i.

    TODO: More efficient implementation with static condensation.
    """

    def __init__(self, space, A):
        """
        Parameters
            space : LagrangeFESpace with degree p
            A     : Stiffness matrix, must be in scipy.sparse.csr_matrix format
        """
        self.space = space
        self.A = A

        self.pre = self.generate_preconditioner()

    def solve(self, r):
        A = self.A
        iter_count = 0
        def callback(xk):
            nonlocal iter_count
            iter_count += 1
        t0 = time.time()
        e = cg(A, r, M = self.pre, callback=callback, rtol=1e-8)[0]
        t1 = time.time()
        print(f"cg iter count: {iter_count}", f"solve time: {t1-t0}")
        return e

    def vertex_preconditioner(self):
        """
        Vertex-based additive Schwarz preconditioner
        """
        A     = self.A
        space = self.space

        p  = space.p
        TD = space.mesh.TD
        NN = space.mesh.number_of_nodes()
        gdof = space.number_of_global_dofs()

        c2d = space.dof.cell_to_dof()
        c2n = space.mesh.entity("cell")

        c2dforv = []
        mindex = bm.multi_index_matrix(p, TD)
        for i in range(TD+1):
            flagi = mindex[:, i] != 0
            c2dforv.append(c2d[:, None, flagi])
        c2dforv = bm.concatenate(c2dforv, axis=1)  # (NC, TD, ndof_per_vertex)

        I = bm.broadcast_to(c2n[:, :, None], c2dforv.shape).reshape(-1)
        J = c2dforv.reshape(-1)
        V = bm.ones_like(I, dtype=bm.float64)
        vertex_dof = csr_matrix((V, (I, J)), shape=(NN, gdof))

        A_diag = [] 
        for i in range(NN):
            start = vertex_dof.indptr[i]
            end = vertex_dof.indptr[i+1]
            # 获得与顶点i相关的所有自由度
            dofs = vertex_dof.indices[start:end]

            I = bm.arange(len(dofs), dtype=bm.int32)
            d = bm.ones(len(dofs), dtype=bm.float64)
            L = csr_matrix((d, (I, dofs)), shape=(len(dofs), gdof))
            A_sub = L @ A @ L.T

            A_diag.append([bm.linalg.inv(A_sub.toarray()), dofs])

        def preconditioner(r):
            r = r.astype(bm.float64)
            e = bm.zeros_like(r)
            for i in range(NN):
                A_i, dofs = A_diag[i]
                e[dofs] += A_i @ r[dofs]
            return e
        return LinearOperator(shape=A.shape, matvec=preconditioner, rmatvec=preconditioner)

    def generate_preconditioner(self):
        A     = self.A
        space = self.space

        space0 = LagrangeFESpace(space.mesh, p=1)
        PI = self.projection(space0)

        A0 = ((PI.T)@A@PI).tocsr()
        mask = bm.abs(A0.data) < 1e-15
        A0.data[mask] = 0
        A0.eliminate_zeros()

        P1 = pyamg.smoothed_aggregation_solver(A0)
        P0 = self.vertex_preconditioner()

        def mix_preconditioner(r1, P0, P1, PI, A):
            r1  = r1.astype(bm.float64)
            e1  = PI@P1.solve(PI.T@r1, maxiter=1, cycle='V')
            e1 += 0.5*P0@r1
            return e1 

        pre_fun = lambda r: mix_preconditioner(r, P0, P1, PI, A)
        pre = LinearOperator(shape=A.shape, matvec=pre_fun, rmatvec=pre_fun)
        return pre

    def projection(self, space0):
        """
        L2 projection from space0 to space1
        @TODO: There exists a more efficient implementation.
        """

        space = self.space
        p = space.p
        q = p + 3

        mesh = space0.mesh

        qf = mesh.quadrature_formula(q, "cell") 
        bcs, ws = qf.get_quadrature_points_and_weights()

        phi  = space.basis(bcs) # (NC, NQ, ldof0)
        phi0 = space0.basis(bcs)

        cm = mesh.entity_measure("cell")
        F  = bm.einsum('cql, cqm, q, c-> clm', phi0, phi, ws, cm)
        M  = bm.einsum('cql, cqm, q, c-> clm', phi, phi, ws, cm)
        Minv = bm.linalg.inv(M)
        F = bm.einsum('clm, cmd->cld', F, Minv)

        c2d0   = space0.cell_to_dof()
        gdof0  = space0.number_of_global_dofs()
        c2d  = space.cell_to_dof()
        gdof = space.number_of_global_dofs()

        I = bm.broadcast_to(c2d[:, None], F.shape).reshape(-1)
        J = bm.broadcast_to(c2d0[..., None], F.shape).reshape(-1)
        data = F.reshape(-1)

        IJ = bm.concatenate([I[None, :], J[None, :]], axis=0)
        unique_IJ, index = bm.unique(IJ, axis=1, return_index=True)

        unique_data = data[index]
        M = csr_matrix((unique_data, (unique_IJ[0], unique_IJ[1])), 
                       shape=(gdof, gdof0), dtype=phi.dtype)
        return M.tocsr()

def test_pre(n: int = 10, p: int = 2, dim=2):
    # 创建PDE模型
    if dim == 2:
        pde = CosCosData() 
        mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=n, ny=n)
    else:
        pde = CosCosCosData()
        mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=n, ny=n, nz=n)
    space = LagrangeFESpace(mesh, p=p)

    bform = BilinearForm(space)
    bform.add_integrator(ScalarDiffusionIntegrator(q=p+3, method='fast'))
    A = bform.assembly()

    lform = LinearForm(space)
    lform.add_integrator(ScalarSourceIntegrator(pde.source, q=p+3))
    b = lform.assembly()

    # 边界条件
    gdof = space.number_of_global_dofs()
    A, b = DirichletBC(space, gd=pde.solution).apply(A, b)

    A  = A.to_scipy()
    uh = space.function()

    print("Constructing ASP preconditioner...")
    Solver = ASPForPoisson(space, A)
    print("Solving linear system with ASP preconditioner...")
    uh[:] = Solver.solve(b)
    print("Solving done.")

    error0 = mesh.error(uh, pde.solution)
    error1 = mesh.error(uh.grad_value, pde.gradient)
    print(f"n = {n}, p = {p}, L2 error: {error0}, H1 error: {error1}")
    return error0, error1

if __name__ == "__main__":
    p = int(sys.argv[1])
    n = int(sys.argv[2])
    dim = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    test_pre(n=n, p=p, dim=dim)













