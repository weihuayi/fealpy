
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
        return self.dof.cell_to_dof(index=index)

    def face_to_dof(self, index: Index=_S) -> TensorLike:
        return self.dof.face_to_dof(index=index)

    def edge_to_dof(self, index=_S):
        return self.dof.edge_to_dof(index=index)

    def is_boundary_dof(self, threshold=None, method=None) -> TensorLike:
        if self.ctype == 'C':
            return self.dof.is_boundary_dof(threshold, method=method)
        else:
            raise RuntimeError("boundary dof is not supported by discontinuous spaces.")

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    def project(self, u: Union[Callable[..., TensorLike], TensorLike],) -> TensorLike:
        """Project a function to the FE function space.
        """
        gdof = self.number_of_global_dofs()
        NC = self.mesh.number_of_cells()
        NN = self.mesh.number_of_nodes()
        assert( NN == gdof )
        assert( u.shape[0] == NC )
        assert( self.p == 1 )
        if u.ndim == 1:
            u = u[:, None]
        cell = self.mesh.entity('cell')
        uh = bm.zeros(gdof, dtype=self.ftype, device=self.device)
        nn = bm.zeros(gdof, dtype=self.itype, device=self.device)
        uh = bm.index_add(uh, cell, u) 
        nn = bm.index_add(nn, cell, 1)

        return self.function(uh/nn)

    def interpolate(self, u: Union[Callable[..., TensorLike], TensorLike],) -> TensorLike:
        assert callable(u)

        if not hasattr(u, 'coordtype') or u.coordtype == 'cartesian':
            ips = self.interpolation_points()
            uI = u(ips)
        elif u.coordtype == 'barycentric': # TODO: 这个结果是不对的 
            TD = self.TD
            p = self.p
            bcs = self.mesh.multi_index_matrix(p, TD)/p
            val = u(bcs)
            cell2dof = self.cell_to_dof()
            uI = bm.zeros(self.number_of_global_dofs(), dtype=self.ftype)
            uI = bm.index_add(uI, cell2dof, val) 
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
            uh = bm.set_at(uh, (..., isDDof), gd[isDDof])
        elif callable(gd):
            gd = gd(ipoints[isDDof])
            if uh is None:
                uh = bm.zeros_like(gd)
            uh = bm.set_at(uh, (..., isDDof), gd)
        else:
            raise TypeError("gd must be a tensor or a callable function")
        
        return uh, isDDof

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
    def cell_basis_on_face(self, bc: TensorLike, eindex: TensorLike) -> TensorLike:
        NLF = self.mesh.number_of_faces_of_cells()
        NF = len(eindex)
        NQ = bc.shape[0]
        ldof = self.number_of_local_dofs('cell')
        result = bm.zeros((NF,NQ,ldof),dtype=self.ftype) 
        face2cell = self.mesh.face_to_cell(eindex) 
        cbcs = self.mesh.update_bcs(bc, 'cell')

        for i in range(NLF): 
            phi = self.basis(cbcs[i])
            tag = bm.where(face2cell[:,2]==i)
            result[tag] = phi
        return result
        
    @barycentric
    def cell_grad_basis_on_face(self, bc: TensorLike, eindex: TensorLike, 
                                isleft = True) -> TensorLike:
        TD = self.mesh.TD  ## 一定是单元的
        NLF = self.mesh.number_of_faces_of_cells()
        NF = len(eindex)
        NQ = bc.shape[0]
        ldof = self.number_of_local_dofs('cell')
        result = bm.zeros((NF,NQ,ldof,TD),dtype=self.ftype) 
        
        cbcs = self.mesh.update_bcs(bc, 'cell')   
        face2cell = self.mesh.face_to_cell(eindex)      
        if isleft:
            c_index  = face2cell[:,0]
            e_local_index = face2cell[:,2]
        else :
            c_index  = face2cell[:,1]
            e_local_index = face2cell[:,3]

        for i in range(NLF):
            gphi = self.grad_basis(cbcs[i], c_index)
            tag = bm.where(e_local_index==i)
            result[tag] = gphi[tag]
        return result
    
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
    
    def prolongation_matrix(self, cdegree=[1]):
        """
        Generate a list of interpolation matrices from lower-order spaces to higher-order spaces,
        from the highest to the lowest.
        
        Parameters:
            cdegree[list]: list of the degree of the needed space,from low space to high space
        
        Returns:
            IM[list]: list of the prolongation matrix,from high space to low space
        """
        assert isinstance(cdegree, list), "cdegree must be a list"
        assert all(isinstance(c, int) for c in cdegree), "All in elements cdegree must be integers"
        assert all(c < self.p for c in cdegree), "All elements in cdegree must be less than self.p"
        assert cdegree == sorted(cdegree), "cdegree must be in ascending order"
        assert self.ctype == 'C'
        p = self.p
        Ps = []
        for c in cdegree[-1::-1]:
            Ps.append(self.mesh.prolongation_matrix(c, p))
            p = c
        return Ps

    
