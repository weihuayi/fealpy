
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
    def cell_basis_on_edge(self, bc: TensorLike, eindex: TensorLike) -> TensorLike:
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
    def cell_grad_basis_on_edge(self, bc: TensorLike, eindex: TensorLike, 
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
    
    def grad_recovery(self, uh: TensorLike, method: str='simple'):
        """

        Notes
        -----

        uh 是线性有限元函数，该程序把 uh 的梯度(分片常数）恢复到分片线性连续空间
        中。

        """
        GD = self.GD
        cell2dof = self.cell_to_dof()
        gdof = self.number_of_global_dofs()
        ldof = self.number_of_local_dofs()
        p = self.p
        bc = self.dof.multiIndex/p
        guh = uh.grad_value(bc)
        guh = guh.swapaxes(0, 1)
        rguh = self.function(dim=GD)

        if method == 'simple':
            deg = bm.bincount(cell2dof.flat, minlength = gdof)
            if GD > 1:
                bm.add.at(rguh, (cell2dof, bm.s_[:]), guh)
            else:
                bm.add.at(rguh, cell2dof, guh)

        elif method == 'area':
            measure = self.mesh.entity_measure('cell')
            ws = bm.einsum('i, j->ij', measure,bm.ones(ldof))
            deg = bm.bincount(cell2dof.flat,weights = ws.flat, minlength = gdof)
            guh = bm.einsum('ij..., i->ij...', guh, measure)
            if GD > 1:
                bm.add.at(rguh, (cell2dof, bm.s_[:]), guh)
            else:
                bm.add.at(rguh, cell2dof, guh)

        elif method == 'distance':
            ipoints = self.interpolation_points()
            bp = self.mesh.entity_barycenter('cell')
            v = bp[:, bm.newaxis, :] - ipoints[cell2dof, :]
            d = bm.sqrt(bm.sum(v**2, axis=-1))
            deg = bm.bincount(cell2dof.flat,weights = d.flat, minlength = gdof)
            guh = bm.einsum('ij..., ij->ij...', guh, d)
            if GD > 1:
                bm.add.at(rguh, (cell2dof, bm.s_[:]), guh)
            else:
                bm.add.at(rguh, cell2dof, guh)

        elif method == 'area_harmonic':
            measure = 1/self.mesh.entity_measure('cell')
            ws = bm.einsum('i, j->ij', measure,bm.ones(ldof))
            deg = bm.bincount(cell2dof.flat,weights = ws.flat, minlength = gdof)
            guh = bm.einsum('ij..., i->ij...', guh, measure)
            if GD > 1:
                bm.add.at(rguh, (cell2dof, bm.s_[:]), guh)
            else:
                bm.add.at(rguh, cell2dof, guh)

        elif method == 'distance_harmonic':
            ipoints = self.interpolation_points()
            bp = self.mesh.entity_barycenter('cell')
            v = bp[:, bm.newaxis, :] - ipoints[cell2dof, :]
            d = 1/bm.sqrt(bm.sum(v**2, axis=-1))
            deg = bm.bincount(cell2dof.flat,weights = d.flat, minlength = gdof)
            guh = bm.einsum('ij..., ij->ij...',guh,d)
            if GD > 1:
                bm.add.at(rguh, (cell2dof, bm.s_[:]), guh)
            else:
                bm.add.at(rguh, cell2dof, guh)
        rguh /= deg.reshape(-1, 1)
        return rguh
