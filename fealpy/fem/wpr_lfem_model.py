      
from typing import Optional, Union
from fealpy.backend import bm

from fealpy.typing import TensorLike
from fealpy.decorator import variantmethod, cartesian

from fealpy.mesh import TriangleMesh, IntervalMesh
from fealpy.functionspace import LagrangeFESpace, functionspace
from fealpy.fem import BilinearForm, LinearForm, DirichletBC, BlockForm, LinearBlockForm
from fealpy.fem import ScalarDiffusionIntegrator, ScalarMassIntegrator, PressWorkIntegrator, CouplingMassIntegrator
from fealpy.model import PDEModelManager, ComputationalModel

from fealpy.mesher import WPRMesher

from fealpy.sparse import spdiags, coo_matrix, csr_matrix, COOTensor, CSRTensor
from fealpy.solver import MGStokes, spsolve, transferP1red, transferP2red, indofP1, indofP2
from fealpy.utils import timer
from fealpy.sparse.ops import bmat

import scipy.sparse as sp
import scipy.sparse.linalg as lg
import time
import gc

class SumOperator:
    def __init__(self, *ops):
        self.ops = ops
        self.shape = ops[0].shape

    def __matmul__(self, x):
        y = 0
        for op in self.ops:
            y = y + (op @ x)
        return y


class LinearOperator:
    def __matmul__(self, x):
        raise NotImplementedError
    
    def __add__(self, other):
        return SumOperator(self, other)

    def __radd__(self, other):
        return self if other == 0 else self.__add__(other)


class KronOperator(LinearOperator):
    def __init__(self, A, B, num=1):
        self.A = A
        self.B = B
        self.num = num
        self.n0, self.m0 = A.shape
        self.n1, self.m1 = B.shape
        self.n = self.m0 * self.m1
        self.shape = (num*self.n0*self.n1, num*self.m0*self.m1)

    def __matmul__(self, x):
        v = bm.copy(x)
        A = self.A
        B = self.B

        if self.num == 3:
            U1 = bm.reshape(v[:self.n], (self.m0, self.m1))
            U2 = bm.reshape(v[self.n:2*self.n], (self.m0, self.m1))
            U3 = bm.reshape(v[2*self.n:3*self.n], (self.m0, self.m1))

            Y1 = A @ U1 @ B
            Y2 = A @ U2 @ B
            Y3 = A @ U3 @ B
            Y = bm.concat([Y1.ravel(), Y2.ravel(), Y3.ravel()], axis=0)
            return Y
        elif self.num == 1:
            X = bm.reshape(v, (self.m0, self.m1))
            Y = A @ X @ B
            Y = Y.ravel()
        return Y


class StokesOperator(LinearOperator):
    def __init__(self, Ax, Mx, Az, Mz, Bx, Bz, Mx_, Mz_):
        self.Ax = Ax
        self.Mx = Mx
        self.Az = Az
        self.Mz = Mz
        self.Bx = Bx
        self.Bz = Bz
        self.Mx_ = Mx_
        self.Mz_ = Mz_
        self.set_up()

    def set_up(self):
        self.n_Ax = self.Ax.shape[0]
        self.n_Mz = self.Mz.shape[0]

        self.n_Bx = self.Bx.shape[0]
        self.m_Bx = self.Bx.shape[1]

        self.n_Mz_ = self.Mz_.shape[0]
        self.m_Mz_ = self.Mz_.shape[1]

        self.n_Mx_ = self.Mx_.shape[0]
        self.m_Mx_ = self.Mx_.shape[1]

        self.n_Bz = self.Bz.shape[0]
        self.m_Bz = self.Bz.shape[1]

        self.n_u0 = self.n_Ax * self.n_Mz
        self.n_p = self.n_Bx * self.n_Mz_
        self.n_A = 3 * self.n_u0 + self.n_p
        self.shape = self.n_A, self.n_A

    def assembly(self):
        pass

    def __matmul__(self, x):
        v = bm.copy(x)

        U1 = bm.reshape(v[:self.n_u0], (self.n_Ax, self.n_Mz))
        U2 = bm.reshape(v[self.n_u0:2*self.n_u0], (self.n_Ax, self.n_Mz))
        U3 = bm.reshape(v[2*self.n_u0:3*self.n_u0], (self.n_Ax, self.n_Mz))

        U4 = bm.reshape(v[:2*self.n_u0], (self.m_Bx, self.m_Mz_))
        U5 = bm.reshape(v[2*self.n_u0:3*self.n_u0], (self.m_Mx_, self.m_Bz))

        P = bm.reshape(v[-self.n_p:], (self.n_Bx, self.n_Mz_))
        
        U1 = bm.to_numpy(U1)
        U2 = bm.to_numpy(U2)
        U3 = bm.to_numpy(U3)
        U4 = bm.to_numpy(U4)
        U5 = bm.to_numpy(U5)
        P = bm.to_numpy(P)

        AU1 = (self.Mz @ (self.Ax @ U1).T).T  + (self.Az @ (self.Mx @ U1).T).T
        AU2 = (self.Mz @ (self.Ax @ U2).T).T  + (self.Az @ (self.Mx @ U2).T).T
        AU3 = (self.Mz @ (self.Ax @ U3).T).T  + (self.Az @ (self.Mx @ U3).T).T

        BP1 = (self.Mz_.T @ (self.Bx.T @ P).T).T
        BP2 = (self.Bz.T @ (self.Mx_.T @ P).T).T
        
        BU1 = (self.Mz_ @ (self.Bx @ U4).T).T
        BU2 = (self.Bz @ (self.Mx_ @ U5).T).T
        
        l1 = bm.concat([bm.tensor(AU1.ravel()), bm.tensor(AU2.ravel())], axis=0) + bm.tensor(BP1.ravel())
        l2 = bm.tensor(AU3.ravel()) + bm.tensor(BP2.ravel())
        l3 = bm.tensor(BU1.ravel()) + bm.tensor(BU2.ravel())

        y = bm.concat([l1, l2, l3], axis=0)

        return y


class A0iOperator(LinearOperator):
    def __init__(self, Ax, Mx, Mz, Az):
        self.Ax = Ax
        self.Mx = Mx
        self.Mz = Mz
        self.Az = Az

        self.m0, self.n0 = Ax.shape
        self.m1, self.n1 = Mz.shape
        self.shape = (self.m0*self.m1, self.n0*self.n1)

    def set_up(self):
        pass

    def assembly(self):
        A0_dense = sp.kron(sp.tril(A=self.Ax, k=-1), self.Mz, format='csr') + \
             sp.kron(sp.tril(A=self.Mx, k=-1), self.Az, format='csr') + \
             sp.kron(sp.diags(self.Ax.diagonal()), sp.tril(A=self.Mz), format='csr') + \
             sp.kron(sp.diags(self.Mx.diagonal()), sp.tril(A=self.Az), format='csr')
        return A0_dense
    
    def __matmul__(self, x):
        v = bm.copy(x)
        X = bm.reshape(v, (self.n0, self.m1))
        Y = self.Ax @ X @ self.Mz + self.Mx @ X @ self.Az
        Y = Y.ravel()
        return Y


class AiOperator(LinearOperator):
    def __init__(self, Ax, Mx, Mz, Az):
        self.Ax = Ax
        self.Mx = Mx
        self.Mz = Mz
        self.Az = Az

        self.m0, self.n0 = Ax.shape
        self.m1, self.n1 = Mz.shape
        self.n_u0 = self.m0*self.m1
        self.shape = (3*self.m0*self.m1, 3*self.n0*self.n1)

    def set_up(self):
        pass

    def assembly(self):
        A0_dense = sp.kron(self.Ax, self.Mz) + sp.kron(self.Mx, self.Az)
        A_dense = sp.block_diag((A0_dense, A0_dense, A0_dense))
        return A_dense
    
    def __matmul__(self, x):
        v = bm.copy(x)
        U1 = bm.reshape(v[:self.n_u0], (self.n0, self.m1))
        U2 = bm.reshape(v[self.n_u0:2*self.n_u0], (self.n0, self.m1))
        U3 = bm.reshape(v[2*self.n_u0:3*self.n_u0], (self.n0, self.m1))

        Y1 = self.Ax @ U1 @ self.Mz + self.Mx @ U1 @ self.Az
        Y2 = self.Ax @ U2 @ self.Mz + self.Mx @ U2 @ self.Az
        Y3 = self.Ax @ U3 @ self.Mz + self.Mx @ U3 @ self.Az
        Y = bm.concat([Y1.ravel(), Y2.ravel(), Y3.ravel()], axis=0)
        return Y


class BiOperator(LinearOperator):
    def __init__(self, Bx, Mx_, Mz_, Bz):
        self.Bx = Bx
        self.Mx_ = Mx_
        self.Mz_ = Mz_
        self.Bz = Bz
        self.Mz_t = Mz_.T
        self.Bzt = Bz.T

        self.n_Bx = self.Bx.shape[0]
        self.m_Bx = self.Bx.shape[1]

        self.n_Mz_ = self.Mz_.shape[0]
        self.m_Mz_ = self.Mz_.shape[1]

        self.n_Mx_ = self.Mx_.shape[0]
        self.m_Mx_ = self.Mx_.shape[1]

        self.n_Bz = self.Bz.shape[0]
        self.m_Bz = self.Bz.shape[1]

        self.m0, self.n0 = Mx_.shape
        self.m1, self.n1 = Bz.shape
        self.n_u0 = self.n0*self.n1
        self.shape = (self.m0*self.m1, 3*self.n0*self.n1)

    def set_up(self):
        pass

    def assembly(self):
        B0 = sp.kron(self.Bx, self.Mz_)
        B1 = sp.kron(self.Mx_, self.Bz)
        B = sp.bmat([[B0, B1]])
        return B

    def __matmul__(self, x):
        v = bm.copy(x)
        U1 = bm.reshape(v[:2*self.n_u0], (self.m_Bx, self.m_Mz_))
        U2 = bm.reshape(v[2*self.n_u0:], (self.m_Mx_, self.m_Bz))

        BU1 = self.Bx @ U1 @ self.Mz_t
        BU2 = self.Mx_ @ U2 @ self.Bzt
        y = BU1.ravel() + BU2.ravel()
        return y


class BtiOperator(LinearOperator):
    def __init__(self, Bxt, Mx_t, Mz_, Bz):
        self.Bxt = Bxt
        self.Mx_t = Mx_t
        self.Mz_ = Mz_
        self.Bz = Bz
        self.Mz_t = Mz_.T
        self.Bzt = Bz.T

        self.n0, self.m0 = Mx_t.shape
        self.m1, self.n1 = Bz.shape
        self.shape = (3*self.n0*self.n1, self.m0*self.m1)

    def set_up(self):
        pass

    def assembly(self):
        B0 = sp.kron(self.Bxt, self.Mz_t)
        B1 = sp.kron(self.Mx_t, self.Bzt)
        B = sp.bmat([[B0], [B1]])
        return B

    def __matmul__(self, x):
        v = bm.copy(x)
        P = bm.reshape(v, (self.m0, self.m1))
        BP1 = self.Bxt @ P @ self.Mz_
        BP2 = self.Mx_t @ P @ self.Bz

        y = bm.concat([BP1.ravel(), BP2.ravel()], axis=0)
        return y


class WPRLFEMModel(ComputationalModel):
    """
    Multigrid solver for Poisson and Stokes-type problems defined on
    tensor-product grids using the Linear Finite Element Method (LFEM).

    Main ideas:
    1. Reduce the matrix size by acting only on interior DoFs, which also 
       reduces the size of interpolation and restriction operators.

    2. Overall strategies:
       Plan I: Use operator-based representation for all components,
               combined with Kronecker-product-based smoothers 
               (optimal performance, but currently lacks full theory).
       Plan II: Use operator-based representation for all components 
                except the smoother (the current implementation).

    3. Storage scheme:
       - Ai:  Operator (level-i stiffness matrix)
       - Bi:  Operator (discrete divergence or coupling operator)
       - Bti: Operator (transpose of Bi)
       - P_u: Operator (velocity interpolation operator)
       - P_p: Operator (pressure interpolation operator)

       Smoother:
       - A0:  Obtained by assembling A0i on the finest level.

       Additional stored 3D matrices for each multigrid level:
       - BB^T
       - tril(BB^T)
       - triu(BB^T)
       - BAB^T
    """
    def __init__(self, options: dict = None):
        self.options = options
        super().__init__(
            pbar_log=options.get('pbar_log', True),
            log_level=options.get('log_level', 'INFO')
        )

        if options is None:
            options = {} 
        
        self.eps = 1e-10
        self.thickness = options.get('thickness', 0.1)

        self.options = options
        self.level = options.get('level', 4)
        self.x0 = options.get('x0', None)
        self.tol = options.get('tol', 1e-8)  

        self.assembly_time = 0
        self.setup_time = 0
        self.initial_assembly_time = 0

    def set_init_mesher(self, mesh:TriangleMesh, imesh: IntervalMesh):
        """
        Set the initial mesh for the simulation.
        
        Parameters:
            mesh: The computational mesh object
        """
        tmesh = mesh
        self.mesh0 = TriangleMesh(tmesh.entity('node'), tmesh.entity('cell'))
        self.mesh1 = TriangleMesh(tmesh.entity('node'), tmesh.entity('cell'))
        tmesh.uniform_refine(self.level-1)
        self.tmesh = tmesh
        self.imesh = imesh

    def set_space_degree(self, p: int=2):
        """
        Set the polynomial degree for function spaces
        """
        self.p = p

    def set_inlet_condition(self)-> None:
        """
        Set the PDE data for the model.
        """
        @cartesian
        def inlet_velocity(p: TensorLike) -> TensorLike:
            """Compute exact solution of velocity."""
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
            result = bm.zeros(p.shape, dtype=bm.float64)
            len = self.options.get('inlet_width', 0.8)
            result[..., 0] = 20*25**2 *(y - (1-0.5*len)) * (1+0.5*len-y) * z * (0.4-z)
            result[..., 1] = bm.array(0.0)
            return result
        
        @cartesian
        def wall_velocity(p: TensorLike) -> TensorLike:
            """Compute exact solution of velocity."""
            x = p[..., 0]
            y = p[..., 1]
            result = bm.zeros(p.shape, dtype=bm.float64)
            result[..., 0] = bm.array(0.0)
            result[..., 1] = bm.array(0.0)
            return result
        
        @cartesian
        def outlet_pressure(p: TensorLike) -> TensorLike:
            """Compute exact solution of velocity."""
            x = p[..., 0]
            y = p[..., 1]
            result = bm.zeros(p.shape[0], dtype=bm.float64)
            result[:] = 0.0
            return result
        
        @cartesian
        def is_inlet_boundary( p: TensorLike) -> TensorLike:
            """Check if point where velocity is defined is on boundary."""
            tag = bm.abs(p[..., 0] - 0.0) < self.eps
            return tag
       
        @cartesian
        def is_outlet_boundary( p: TensorLike) -> TensorLike:
            """Check if point where pressure is defined is on boundary."""
            tag = bm.abs(p[..., 0] - 6.0) < self.eps
            return tag

        @cartesian
        def is_wall_boundary(p: TensorLike) -> TensorLike:
            """Check if point where velocity is defined is on boundary."""
            len =  self.options.get('gap_len', 0.1)
            inlet_len = self.options.get('inlet_width', 0.5)
            
            bd0 = bm.array([
                            [0.0, 1-0.5*inlet_len], [0.5, 1-0.5*inlet_len], [0.0, 1+0.5*inlet_len], [0.5, 1+0.5*inlet_len],
                            [0.5, 1-0.5*inlet_len], [0.5, 0.00], [0.5, 1+0.5*inlet_len], [0.5, 2.00],

                            [2.5, 0], [2.5, len], [2.5, len], [2.6, len], [2.6, len], [2.6, 0],
                            [4.5, 0], [4.5, len], [4.5, len], [4.6, len], [4.6, len], [4.6, 0],

                            [5.5, 0.00], [5.5, 1-0.5*inlet_len], [5.5, 1-0.5*inlet_len], [6.0, 1-0.5*inlet_len],
                            [5.5, 1+0.5*inlet_len], [5.5, 2.00], [5.5, 1+0.5*inlet_len], [6.0, 1+0.5*inlet_len],

                            [3.5, 2-len], [3.6, 2-len], [3.5, 2-len], [3.5, 2], [3.6, 2-len], [3.6, 2],
                            [1.5, 2-len], [1.6, 2-len], [1.5, 2-len], [1.5, 2], [1.6, 2-len], [1.6, 2],
                           ])
            cond0 = self.is_lateral_boundary(p, bd0)
            cond1 = (bm.abs(p[..., 1]) < self.eps) | (bm.abs(p[..., 1] - 2.0) < self.eps)
            return cond0 | cond1
        
        @cartesian
        def is_top_or_bottom(p: TensorLike) -> TensorLike:
            """Check if point where velocity is defined is on top or bottom boundary."""
            atol = 1e-12
            thickness = self.thickness
            cond = (bm.abs(p[:, -1]) < atol) | (bm.abs(p[:, -1] - thickness) < atol)
            return cond
                
        self.inlet_velocity = inlet_velocity
        self.wall_velocity = wall_velocity
        self.outlet_pressure = outlet_pressure

        self.is_inlet_boundary = is_inlet_boundary
        self.is_outlet_boundary = is_outlet_boundary
        self.is_wall_boundary = is_wall_boundary
        self.is_top_or_bottom = is_top_or_bottom

    def is_lateral_boundary(self, p: TensorLike, bd: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        atol = 1e-12
        v0 = p[:, None, :-1] - bd[None, 0::2, :] # (NN, NI, 2)
        v1 = p[:, None, :-1] - bd[None, 1::2, :] # (NN, NI, 2)

        cross = v0[..., 0]*v1[..., 1] - v0[..., 1]*v1[..., 0] # (NN, NI)
        dot = bm.einsum('ijk,ijk->ij', v0, v1) # (NN, NI)
        cond = (bm.abs(cross) < atol) & (dot < atol)
        return bm.any(cond, axis=1)
    
    @cartesian
    def is_velocity_boundary(self, p: TensorLike, dim=3) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        inlet = self.is_inlet_boundary(p)
        wall = self.is_wall_boundary(p)
        top_or_bottom = self.is_top_or_bottom(p)
        if dim == 2:
            return inlet | wall
        return inlet | wall | top_or_bottom
    
    @cartesian
    def is_pressure_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        return self.is_outlet_boundary(p)

    @cartesian
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
        """Optional: prescribed velocity on boundary, if needed explicitly."""
        inlet = self.inlet_velocity(p)
        is_inlet = self.is_inlet_boundary(p)
        
        result = bm.zeros_like(p, dtype=p.dtype)
        result[is_inlet] = inlet[is_inlet]

        return result
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        """Optional: prescribed pressure on boundary (usually for stability)."""
        outlet = self.outlet_pressure(p)
        is_outlet = self.is_outlet_boundary(p)
        result = bm.zeros_like(p[..., 0], dtype=p.dtype)
        result[is_outlet] = outlet[is_outlet]
        return result

    @variantmethod
    def linear_system(self):
        """
        Assemble the linear system for the Stokes equations.
        """
        from fealpy.mesh import TensorPrismMesh
        self.mesh = TensorPrismMesh(self.tmesh, self.imesh)
        
        self.uspace = functionspace(self.mesh, ('Lagrange', 2), shape=(3, -1))
        self.pspace = functionspace(self.mesh, ('Lagrange', 1))
        
        self.int_space0 = LagrangeFESpace(self.imesh, p=1)
        self.int_space1 = LagrangeFESpace(self.imesh, p=2)
        self.tri_space0 = LagrangeFESpace(self.tmesh, p=1)
        self.tri_space1 = LagrangeFESpace(self.tmesh, p=2)

        form00 = BilinearForm(self.tri_space1)
        form00.add_integrator(ScalarDiffusionIntegrator())
        Ax = form00.assembly().to_scipy()

        form01 = BilinearForm(self.tri_space1)
        form01.add_integrator(ScalarMassIntegrator())
        Mx = form01.assembly().to_scipy()

        form02 = BilinearForm(self.int_space1)
        form02.add_integrator(ScalarDiffusionIntegrator())
        Az = form02.assembly().to_scipy()

        form03 = BilinearForm(self.int_space1)
        form03.add_integrator(ScalarMassIntegrator())
        Mz = form03.assembly().to_scipy()

        self.uspace2d = functionspace(self.tmesh, ('Lagrange', 2), shape=(2, -1))
        self.pspace2d = functionspace(self.tmesh, ('Lagrange', 1))

        form10 = BilinearForm((self.pspace2d, self.uspace2d))
        form10.add_integrator(PressWorkIntegrator(coef=-1.0))
        Bx = form10.assembly().to_scipy().T

        form11 = BilinearForm((self.int_space0, self.int_space1))
        form11.add_integrator(CouplingMassIntegrator())
        Mz_ = form11.assembly().to_scipy().T

        self.uspace1d = functionspace(self.imesh, ('Lagrange', 2), shape=(1, -1))
        self.pspace1d = functionspace(self.imesh, ('Lagrange', 1))

        form12 = BilinearForm((self.pspace1d, self.uspace1d))
        form12.add_integrator(PressWorkIntegrator(coef=-1.0))
        Bz = form12.assembly().to_scipy().T

        form13 = BilinearForm((self.tri_space0, self.tri_space1))
        form13.add_integrator(CouplingMassIntegrator())
        Mx_ = form13.assembly().to_scipy().T
        
        self.ugdof = Ax.shape[0]*Mz.shape[0]
        self.total_dof = Ax.shape[0]*Mz.shape[0]*3+Bx.shape[1]*Mz_.shape[1]
        print(f'自由度个数: {Ax.shape[0]*Mz.shape[0]*3+Bx.shape[1]*Mz_.shape[1]}')
        op = StokesOperator(Ax, Mx, Az, Mz, Bx, Bz, Mx_, Mz_)
       
        # A1 = sp.kron(Ax, Mz) + \
        #      sp.kron(Mx, Az)
        # B0 = sp.kron(Bx, Mz_)
        # B1 = sp.kron(Mx_, Bz)
        
        # A0 = sp.block_diag((A1, A1, A1))
        # B = sp.bmat([[B0, B1]])
        # A = sp.bmat([[A0, B.T],
        #              [B, None]])

        # from fealpy.sparse import COOTensor
        # A = COOTensor(
        #     indices=bm.stack([A.row, A.col], axis=0),
        #     values=A.data,
        #     spshape=A.shape
        # )
        A = None
        self.n_A = op.n_A
        self.n_p = op.n_p
        self.x0 = bm.zeros((self.n_A,), dtype=bm.float64)
        F = bm.zeros((self.n_A,), dtype=bm.float64)
        return op, A, F
    
    def boundary_dof_index(self):
        isDDof0 = self.tmesh.boundary_node_flag()
        isDDof1 = self.tri_space1.is_boundary_dof()
        isDDof2 = self.imesh.boundary_face_flag()
        igdof = self.int_space1.number_of_global_dofs()
        isDDof3 = bm.zeros((igdof, ), dtype=bm.bool)
        bm.set_at(isDDof3, bm.arange(len(isDDof2)), isDDof2)

        bd_dof0 = ~((~isDDof1[:, None]) * (~isDDof3[None, :])).ravel()
        bd_dof1 = ~((~isDDof0[:, None]) * (~isDDof2[None, :])).ravel()

        return (bd_dof1, bd_dof0)

    def interpolation_points(self):
        ipoint0 = self.imesh.interpolation_points(p=1)
        ipoint1 = self.imesh.interpolation_points(p=2)
        ipoint2 = self.tmesh.interpolation_points(p=1)
        ipoint3 = self.tmesh.interpolation_points(p=2)
        
        p0 = bm.concat([bm.repeat(ipoint2, ipoint0.shape[0], axis=0), 
                          bm.tile(ipoint0.T, (ipoint2.shape[0],)).T], axis=1)
        p1 = bm.concat([bm.repeat(ipoint3, ipoint1.shape[0], axis=0), 
                          bm.tile(ipoint1.T, (ipoint3.shape[0],)).T], axis=1)
        
        return (p0, p1)

    def apply_bc(self, op: StokesOperator, F):
        uh = self.x0
        gd = (self.velocity_dirichlet, self.pressure_dirichlet)
        threshold = (self.is_velocity_boundary, self.is_pressure_boundary)
        
        dofs = self.boundary_dof_index()
        points = self.interpolation_points() # (2000w, 3) ~ 500 MB
        basic = [3*len(points[1]), 0]
        BdDof = []

        for i in range(2):
            index_dof = bm.arange(len(points[i]))[dofs[i]] + basic[i]
            # ipoints: (NI， 3), 边界插值点坐标, 
            bd_point = points[i][dofs[i]] 
            # flag: (NI,), 判断边界点是否属于某类边界
            flag = threshold[1-i](bd_point)
            index_dof = index_dof[flag]
            val = gd[1-i](bd_point[flag])
            if i == 1:
                index_dof = bm.concat([index_dof, index_dof + len(points[1]), 
                                    index_dof + 2*len(points[1])], axis=0)
                val = val.T.reshape(-1)

            BdDof.append(index_dof)
            isBdDof = bm.zeros(self.n_A, dtype=bm.bool)
            isBdDof = bm.set_at(isBdDof, index_dof, True)
            uh = bm.set_at(uh, (..., isBdDof), val)
        BdDof = bm.concat([BdDof[1], BdDof[0]], axis=0)
        F = F - op @ uh # 5000w ~ 400MB
        F = bm.set_at(F, BdDof, uh[BdDof])

        # Fixdof
        flag = self.imesh.boundary_face_flag()
        igdof = self.int_space1.number_of_global_dofs()
        isDDof = bm.zeros((igdof, ), dtype=bm.bool)
        bm.set_at(isDDof, bm.arange(len(flag)), flag)
        
        inflag_uz = ~isDDof
        inflag_u = indofP2(self.tmesh, threshold=self.is_velocity_boundary, tensor_mesh=True)
        inflag_p = indofP1(self.tmesh, threshold=self.is_pressure_boundary, tensor_mesh=True)
        inflag_u = bm.to_numpy(inflag_u)
        Biginflag_u = bm.to_numpy(bm.concat([inflag_u, inflag_u], axis=0))
        inflag_uz = bm.to_numpy(inflag_uz)

        op.Ax = op.Ax[inflag_u][:,inflag_u]
        op.Mx = op.Mx[inflag_u][:,inflag_u]
        op.Az = op.Az[inflag_uz][:,inflag_uz]
        op.Mz = op.Mz[inflag_uz][:,inflag_uz]

        if inflag_p is not None:
            inflag_p = bm.to_numpy(inflag_p)
            op.Bx = op.Bx[inflag_p][:,Biginflag_u]
            op.Mx_ = op.Mx_[inflag_p][:,inflag_u]
            op.Mz_ = op.Mz_[:,inflag_uz]
            op.Bz = op.Bz[:,inflag_uz]
        
        op.set_up()

        return op, F, BdDof

    def setup(self, op: StokesOperator):
        """Compute restriction and interpolation operators.
        """
        Ax = op.Ax
        Mx = op.Mx
        Az = op.Az
        Mz = op.Mz
        Bx = op.Bx
        Bz = op.Bz
        Mx_ = op.Mx_
        Mz_ = op.Mz_

        level = self.level
        Axi = [None] * level
        Mxi = [None] * level
        Bxi = [None] * level
        Mx_i = [None] * level
        
        Axi[-1] = Ax
        Mxi[-1] = Mx
        Bxi[-1] = Bx
        Mx_i[-1] = Mx_
        
        Nu = bm.zeros((level,), dtype=bm.int32)
        Np = bm.zeros((level,), dtype=bm.int32)
        
        # Compute Pro and Res of u and p.
        Pro_p = transferP1red(self.mesh0, self.level, self.is_pressure_boundary)
        Pro_u = transferP2red(self.mesh1, self.level, self.is_velocity_boundary, tensor_mesh=True)
        
        for j in range(level - 1, 0, -1):
            Axi[j-1] = Pro_u[j-1].T @ Axi[j] @ Pro_u[j-1]
            Mxi[j-1] = Pro_u[j-1].T @ Mxi[j] @ Pro_u[j-1]
            Bxi[j-1] = Pro_p[j-1].T @ Bxi[j] @ sp.block_diag([Pro_u[j-1],Pro_u[j-1]])
            Mx_i[j-1] = Pro_p[j-1].T @ Mx_i[j] @ Pro_u[j-1]

        self.P_u = [None] * (level-1)
        self.P_p = [None] * (level-1)
        self.R_u = [None] * (level-1)
        self.R_p = [None] * (level-1)

        self.auxMat = [None] * level
        self.A0i = [None] * level
        self.Ai = [None] * level
        self.Bi = [None] * level
        self.Bti = [None] * level
        self.bigAi = [None] * level

        Iz2 = spdiags(bm.ones((op.n_Mz,)), 0, op.n_Mz, op.n_Mz).to_scipy()
        Iz1 = spdiags(bm.ones((op.n_Mz_,)), 0, op.n_Mz_, op.n_Mz_).to_scipy()

        for j in range(self.level):
            self.A0i[j] = A0iOperator(Axi[j], Mxi[j], Mz, Az)
            self.Ai[j] = AiOperator(Axi[j], Mxi[j], Mz, Az)
            self.Bi[j] = BiOperator(Bxi[j], Mx_i[j], Mz_, Bz)
            self.Bti[j] = BtiOperator(Bxi[j].T, Mx_i[j].T, Mz_, Bz)
            Nu[j] = self.A0i[j].shape[0]
            Np[j] = self.Bi[j].shape[0]
            # bigAi[j] = (sp.bmat([[Ai[j], Bi[j].T],[Bi[j], None]]).tocsr())
            if j < self.level - 1:
                self.P_u[j] = KronOperator(Pro_u[j], Iz2, num=3)
                self.P_p[j] = KronOperator(Pro_p[j], Iz1)
                self.R_u[j] = KronOperator(Pro_u[j].T, Iz2, num=3)
                self.R_p[j] = KronOperator(Pro_p[j].T, Iz1)
            
            if j > 0:
                BBt = sp.kron(Bxi[j]@Bxi[j].T, Mz_@Mz_.T) + sp.kron(Mx_i[j]@Mx_i[j].T, Bz@Bz.T)
                Sp = sp.tril(BBt).tocsr()
                Spt = sp.triu(BBt).tocsr()
                DSp = sp.diags_array(1/BBt.diagonal())

                self.auxMat[j] = {
                    'Bt': self.Bti[j],
                    'BBt': BBt,
                    'Spt': Spt,
                    'Sp': Sp,
                    'invSpt': Spt @ DSp,
                    'invSp': Sp @ DSp
                }

                self.auxMat[j]['BABt'] = sp.kron(Bxi[j]@sp.block_diag((Axi[j], Axi[j]))@Bxi[j].T, Mz_@Mz@Mz_.T) \
                     + sp.kron(Bxi[j]@sp.block_diag((Mxi[j], Mxi[j]))@Bxi[j].T, Mz_@Az@Mz_.T) \
                     + sp.kron(Mx_i[j]@Axi[j]@Mx_i[j].T, Bz@Mz@Bz.T) \
                     + sp.kron(Mx_i[j]@Mxi[j]@Mx_i[j].T, Bz@Az@Bz.T)
                self.auxMat[j]['Su0'] = self.A0i[j].assembly()
        
        self.Nu = Nu
        self.Np = Np
        Ai = self.from_scipy(self.Ai[0].assembly()).tocsr()
        Bti = self.from_scipy(self.Bti[0].assembly()).tocsr()
        Bi = self.from_scipy(self.Bi[0].assembly()).tocsr()
        bigAi = bmat([[Ai, Bti],[Bi, None]])
        self.bigAi = bigAi

    def from_scipy(self, M):
        row = bm.from_numpy(M.row)
        col = bm.from_numpy(M.col)
        data = bm.from_numpy(M.data)
        M = COOTensor(indices=bm.stack([row, col], axis=0),
                        values=data, spshape=M.shape)
        
        return M
  
    @variantmethod('direct')
    def solve(self, op: StokesOperator, F, solver='mumps'):
        """
        Solve the linear system using direct method.
        """
        from fealpy.solver import bicgstab, minres, gmres, cg
        x, info = minres(op, F, atol=1e-8, rtol=1e-8)
        # x, info = cg(op, F, returninfo=True)
        print(info)
        return x

    @solve.register('mg')
    def solve(self, op: StokesOperator, F):
        self.logger.info(f'Step 4. setup 完成\n')

        Solver = MGStokes(self.Ai, self.Bi, self.Bti, self.bigAi,
                        self.P_u, self.R_u, self.P_p, self.R_p,
                        self.Nu, self.Np, self.level, 
                        self.auxMat, self.options)
        
        bigu = Solver.solve(op, F)
        return bigu

    @solve.register('amg')
    def solve(self, A, F):
        raise NotImplementedError("AMG solver not yet implemented.")

    def run(self):
        import time
        start = time.time()
        op0, A, F = self.linear_system()
        self.logger.info(f'Step 1. 完成初步线性系统组装\n')

        import time
        start = time.time()
        self.solver = 'mg'
        # self.solver = 'direct'
        if self.solver == 'direct':
            BC = DirichletBC(
                (self.uspace, self.pspace),
                gd=(self.velocity_dirichlet, self.pressure_dirichlet),
                threshold=(self.is_velocity_boundary, self.is_pressure_boundary),
                method='interp'
            )
            A, F2 = BC.apply(A, F)
            self.logger.info(f'Step 2. 完成边界自由度处理\n')
            self.logger.info(f'Step 3. 开始使用直接法求解\n')
            tmr = timer()
            next(tmr)
            x = spsolve(A, F2)
            print(x.max(), x.min())
            tmr.send(f'求解器时间')
            next(tmr)
            
        elif self.solver == 'mg':           
            op, F1, BdDof = self.apply_bc(op0, bm.copy(F))
            del op0
            gc.collect()
            self.logger.info(f'Step 2. 完成边界自由度处理\n')
            bd_flag = bm.zeros((len(F),), dtype=bm.bool)
            bm.set_at(bd_flag, BdDof, True)
            self.logger.info(f'Step 3. 开始多重网格setup阶段\n')
            self.initial_assembly_time += time.time() - start
            # initial set up
            start = time.time()
            self.setup(op)
            self.setup_time += time.time() - start
            x_in = self.solve['mg'](op, F1[~bd_flag])
            x = bm.set_at(F1, ~bd_flag, x_in)
            print(x.max(), x.min())
        uh = x[:3*self.ugdof]
        ph = x[3*self.ugdof:]
        self.post_process(uh ,ph)
        return uh, ph
    
    def error(self):
        err = bm.sqrt(bm.mean((self.pde.solution(self.node) - self.uh)**2))
        return err
    
    def post_process(self, uh, ph):
        iNN = self.imesh.number_of_nodes()
        tNN = self.tmesh.number_of_nodes()
        tgdof = self.tmesh.number_of_global_ipoints(p=2)
        igdof = self.imesh.number_of_global_ipoints(p=2)
        gdof = tgdof * igdof
        idx = bm.arange(gdof).reshape(tgdof, -1)[:tNN, :iNN].ravel()

        self.mesh.nodedata['ph'] = ph
        self.mesh.nodedata['uh'] = uh.reshape(3,-1).T[idx,:]
        
        self.mesh.to_vtk('dld_prism_chip.vtu')


    