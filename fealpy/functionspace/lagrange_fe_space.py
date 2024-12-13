
from typing import Optional, TypeVar, Union, Generic, Callable
from ..typing import TensorLike, Index, _S, Threshold

from ..backend import TensorLike
from ..backend import backend_manager as bm
from ..mesh.mesh_base import Mesh
from .space import FunctionSpace
from .dofs import LinearMeshCFEDof, LinearMeshDFEDof
from .function import Function
from fealpy.decorator import barycentric, cartesian


_MT = TypeVar('_MT', bound=Mesh)


class LagrangeFESpace(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh: _MT, p: int=1, ctype='C'):
        self.mesh = mesh
        self.p = p

        assert ctype in {'C', 'D'}
        self.ctype = ctype # 空间连续性类型

        if ctype == 'C':
            self.dof = LinearMeshCFEDof(mesh, p)
        elif ctype == 'D':
            self.dof = LinearMeshDFEDof(mesh, p)
        else:
            raise ValueError(f"Unknown type: {ctype}")

        self.ftype = mesh.ftype
        self.itype = mesh.itype
        # self.multi_index_matrix = mesh.multi_index_matrix(p,2)

        #TODO:JAX
        self.device = mesh.device
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

    def __str__(self):
        return "Lagrange finite element space on linear mesh!"

    def number_of_local_dofs(self, doftype='cell') -> int:
        return self.dof.number_of_local_dofs(doftype=doftype)

    def number_of_global_dofs(self) -> int:
        return self.dof.number_of_global_dofs()

    def interpolation_points(self) -> TensorLike:
        return self.dof.interpolation_points()

    def cell_to_dof(self, index: Index=_S) -> TensorLike:
        return self.dof.cell_to_dof()[index]

    def face_to_dof(self, index: Index=_S) -> TensorLike:
        return self.dof.face_to_dof()[index]

    def edge_to_dof(self, index=_S):
        return self.dof.edge_to_dof()[index]

    def is_boundary_dof(self, threshold=None, method=None) -> TensorLike:
        if self.ctype == 'C':
            return self.dof.is_boundary_dof(threshold, method=method)
        else:
            raise RuntimeError("boundary dof is not supported by discontinuous spaces.")

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    def interpolate(self, u: Union[Callable[..., TensorLike], TensorLike],) -> TensorLike:
        assert callable(u)

        if not hasattr(u, 'coordtype'):
            ips = self.interpolation_points()
            uI = u(ips)
        else:
            if u.coordtype == 'cartesian':
                ips = self.interpolation_points()
                uI = u(ips)
            elif u.coordtype == 'barycentric':
                TD = self.TD
                p = self.p
                bcs = self.mesh.multi_index_matrix(p, TD)/p
                uI = u(bcs)
        return self.function(uI)

    def boundary_interpolate(self,
            gd: Union[Callable, int, float, TensorLike],
            uh: Optional[TensorLike] = None,
            *, threshold: Optional[Threshold]=None, method=None) -> TensorLike:
        """Set the first type (Dirichlet) boundary conditions.

        Parameters:
            gd: boundary condition function or value (can be a callable, int, float, TensorLike).
            uh: TensorLike, FE function uh .
            threshold: optional, threshold for determining boundary degrees of freedom (default: None).

        Returns:
            TensorLike: a bool array indicating the boundary degrees of freedom.

        This function sets the Dirichlet boundary conditions for the FE function `uh`. It supports
        different types for the boundary condition `gd`, such as a function, a scalar, or a array.
        """
        ipoints = self.interpolation_points() # TODO: 直接获取过滤后的插值点
        isDDof = self.is_boundary_dof(threshold=threshold, method='interp')
        if bm.is_tensor(gd):
            assert len(gd) == self.number_of_global_dofs()
            if uh is None:
                uh = bm.zeros_like(gd)
            uh[isDDof] = gd[isDDof] 
            return uh,isDDof 
        if callable(gd):
            gd = gd(ipoints[isDDof])
        if uh is None:
            uh = self.function()
        uh[:] = bm.set_at(uh[:], (..., isDDof), gd)
        
        return self.function(uh), isDDof

    set_dirichlet_bc = boundary_interpolate

    def basis(self, bc: TensorLike, index: Index=_S):
        phi = self.mesh.shape_function(bc, self.p, index=index)
        return phi[None, ...] # (NC, NQ, LDOF)

    face_basis = basis
    edge_basis = basis

    def grad_basis(self, bc: TensorLike, index: Index=_S, variable='x'):
        return self.mesh.grad_shape_function(bc, self.p, index=index, variables=variable)

    def hess_basis(self, bc: TensorLike, index: Index=_S, variable='x'):
        return self.mesh.hess_shape_function(bc, self.p, index=index, variables=variable)

    @barycentric
    def cell_basis_on_edge(self, bc: TensorLike, lidx: TensorLike,
                           direction=True) -> TensorLike:
        TD = self.mesh.TD  ## 一定是单元的
        NLF = self.mesh.number_of_faces_of_cells()
        NF = self.mesh.number_of_faces()
        NQ = bc.shape[0]
        ldof = self.number_of_local_dofs('cell')
        result = bm.zeros((NF,NQ,ldof,TD),dtype=self.ftype) 
        face2cell = self.mesh.face_to_cell() 
        cell2face = self.mesh.cell_to_face()
        
        for i in range(NLF):
            cbcs = bm.insert(bc, i, 0.0, axis=-1)
            phi = self.basis(cbcs)
            if direction: ##左边单元
                tag = bm.where(face2cell[:,2]==i)
                result[tag] = phi[face2cell[tag,0]]
            else: ##右边单元
                tag = bm.where(face2cell[:,3]==i)
                result[tag] = phi[face2cell[tag,1]]
        ii = cell2face[...,lidx]
        return result[ii]
        
        """
        @brief Return the basis value of cells on points of edges.

        @param bc: TensorLike. Barycentric coordinates of points on edges, with shape [NE, 2].
        @param lidx: TensorLike. The local index of edges, with shape [NE, ].
        @param direction: bool. True for the default direction of the edge, False for the opposite direction.

        @return: Basis with shape [NE, NQ, Dofs].
        """
        '''
        if bc.shape[-1] != 2:
            raise ValueError('The shape of bc should be [NE, 2].')
        if lidx.ndim != 1:
            raise ValueError('lidx is expected to be 1-dimensional.')

        NE = len(lidx)
        nmap = bm.array([1, 2, 0])
        pmap = bm.array([2, 0, 1])
        shape = (NE, ) + bc.shape[:-1] + (3, )
        bcs = bm.zeros(shape, dtype=self.mesh.ftype)
        idx = bm.arange(NE)

        if direction:
            bcs[idx, ..., nmap[lidx]] = bc[..., 0]
            bcs[idx, ..., pmap[lidx]] = bc[..., 1]
        else:
            bcs[idx, ..., nmap[lidx]] = bc[..., 1]
            bcs[idx, ..., pmap[lidx]] = bc[..., 0]
        return self.mesh.shape_function(bcs, p=self.p)
        '''
    @barycentric
    def cell_grad_basis_on_edge(self, bc: TensorLike, index: TensorLike, lidx: TensorLike,
                                direction=True) -> TensorLike:
        TD = self.mesh.TD  ## 一定是单元的
        NLF = self.mesh.number_of_faces_of_cells()
        NF = self.mesh.number_of_faces()
        NQ = bc.shape[0]
        ldof = self.number_of_local_dofs('cell')
        result = bm.zeros((NF,NQ,ldof,TD),dtype=self.ftype) 
        face2cell = self.mesh.face_to_cell() 
        cell2face = self.mesh.cell_to_face()
        
        for i in range(NLF):
            cbcs = bm.insert(bc, i, 0.0, axis=-1)
            ggphi = self.grad_basis(cbcs)
            if direction: ##左边单元
                tag = bm.where(face2cell[:,2]==i)
                result[tag] = ggphi[face2cell[tag,0]]
            else: ##右边单元
                tag = bm.where(face2cell[:,3]==i)
                result[tag] = ggphi[face2cell[tag,1]]
        ii = cell2face[index,lidx]
        return result[ii]
    ''' 
    @barycentric
    def edge_grad_basis(self, bc, index, lidx, direction=True):
        """

        Notes
        -----
            bc：边上的一组重心坐标积分点
            index: 边所在的单元编号
            lidx: 边在该单元的局部编号
            direction: True 表示边的方向和单元的逆时针方向一致，False 表示不一致

            计算基函数梯度在单元边上积分点的值.

            这里要把边上的低维的积分点转化为高维的积分点.

        TODO
        ----
            二维和三维统一？
            有没有更好处理办法？

        """
        NE = len(index)
        nmap = np.array([1, 2, 0])
        pmap = np.array([2, 0, 1])
        shape = (NE, ) + bc.shape[0:-1] + (3, )
        bcs = np.zeros(shape, dtype=self.mesh.ftype)  # (NE, 3) or (NE, NQ, 3)
        idx = np.arange(NE)
        if direction:
            bcs[idx, ..., nmap[lidx]] = bc[..., 0]
            bcs[idx, ..., pmap[lidx]] = bc[..., 1]
        else:
            bcs[idx, ..., nmap[lidx]] = bc[..., 1]
            bcs[idx, ..., pmap[lidx]] = bc[..., 0]

        p = self.p   # the degree of polynomial basis function
        TD = self.TD
        multiIndex = self.mesh.multi_index_matrix(p, TD)

        c = np.arange(1, p+1, dtype=self.itype)
        P = 1.0/np.multiply.accumulate(c)

        t = np.arange(0, p)
        shape = bcs.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bcs[..., np.newaxis, :] - t.reshape(-1, 1)

        FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
        FF[..., range(p), range(p)] = p
        np.cumprod(FF, axis=-2, out=FF)
        F = np.zeros(shape, dtype=self.ftype)
        F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
        F[..., 1:, :] *= P.reshape(-1, 1)

        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)

        Q = A[..., multiIndex, range(TD+1)]
        M = F[..., multiIndex, range(TD+1)]
        ldof = self.number_of_local_dofs()
        shape = bcs.shape[:-1]+(ldof, TD+1)
        R = np.zeros(shape, dtype=self.ftype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)

        Dlambda = self.mesh.grad_lambda()
        gphi = np.einsum('k...ij, kjm->k...im', R, Dlambda[index, :, :])
        return gphi
    '''
    @barycentric
    def value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike: 
        if isinstance(bc, tuple):
            TD = len(bc)
        else :
            TD = bc.shape[-1] - 1
        phi = self.basis(bc, index=index)
        e2dof = self.dof.entity_to_dof(TD, index=index)
        val = bm.einsum('cql, ...cl -> ...cq', phi, uh[..., e2dof])
        return val

    @barycentric
    def grad_value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike:
        if isinstance(bc, tuple):
            TD = len(bc)
        else :
            TD = bc.shape[-1] - 1
        gphi = self.grad_basis(bc, index=index)
        e2dof = self.dof.entity_to_dof(TD, index=index)
        val = bm.einsum('cilm, cl -> cim', gphi, uh[e2dof])
        return val
