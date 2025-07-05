
from typing import Optional, TypeVar, Union, Generic, Callable
from fealpy.typing import TensorLike, Index, _S, Threshold

from fealpy.backend import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.mesh.mesh_base import Mesh
from fealpy.functionspace import FunctionSpace
from fealpy.functionspace.function import Function
from fealpy.functionspace.functional import symmetry_span_array, symmetry_index
from fealpy.decorator import barycentric, cartesian

import time

def number_of_multiindex(p, d):
    if d == 1:
        return p+1
    elif d == 2:
        return (p+1)*(p+2)//2
    elif d == 3:
        return (p+1)*(p+2)*(p+3)//6

def multiindex_to_number(a):
    d = a.shape[1] - 1
    if d==1:
        return a[:, 1]
    elif d==2:
        a1 = a[:, 1] + a[:, 2]
        a2 = a[:, 2]
        return a1*(1+a1)//2 + a2 
    elif d==3:
        a1 = a[:, 1] + a[:, 2] + a[:, 3]
        a2 = a[:, 2] + a[:, 3]
        a3 = a[:, 3]
        return a1*(1+a1)*(2+a1)//6 + a2*(1+a2)//2 + a3

class TensorDofsOnSubsimplex():
    def __init__(self, dofs : list, subsimplex : list):
        """
        dofs: list of tuple (alpha, I), alpha is the multi-index, I is the
              tensor index.
        """
        self.dof_scalar = bm.array([dof[0] for dof in dofs], dtype=bm.int32)
        self.dof_tensor = bm.array([dof[1] for dof in dofs], dtype=bm.int32)

        self.subsimplex = subsimplex

        self.dof2num = self._get_dof_to_num()

    def __getitem__(self, idx):
        return self.dof_scalar[idx], self.dof_tensor[idx]

    def __len__(self):
        return self.dof_scalar.shape[0]

    def _get_dof_to_num(self):
        alpha = self.dof_scalar
        I     = self.dof_tensor
        ldof  = number_of_multiindex(bm.sum(alpha[0]), alpha.shape[1]-1)
        idx = multiindex_to_number(alpha) + I*ldof

        nummap = bm.zeros((idx.max()+1,), dtype=alpha.dtype)
        nummap[idx] = bm.arange(len(idx), dtype=alpha.dtype)
        return nummap

    def permute_to_order(self, perm):
        alpha = self.dof_scalar.copy()
        alpha[:, self.subsimplex] = alpha[:, self.subsimplex][:, perm]

        I     = self.dof_tensor
        ldof  = number_of_multiindex(bm.sum(alpha[0]), alpha.shape[1]-1)
        idx = multiindex_to_number(alpha) + I*ldof
        return self.dof2num[idx]

class HuZhangFECellDof2d():
    def __init__(self, mesh : Mesh, p: int):
        self.p = p
        self.mesh = mesh
        self.TD = mesh.top_dimension() 

        self._get_simplex()
        self.boundary_dofs, self.internal_dofs = self.dof_classfication()

    def _get_simplex(self):
        TD = self.TD 
        mesh = self.mesh
        
        localnode = bm.array([[0], [1], [2]], dtype=mesh.itype)
        localcell = bm.array([[0, 1, 2]], dtype=mesh.itype)
        self.subsimplex = [localnode, mesh.localEdge, localcell]


        dual = lambda alpha : [i for i in range(self.TD+1) if i not in alpha]
        self.dual_subsimplex = [[dual(f) for f in ssixi] for ssixi in self.subsimplex]

    def dof_classfication(self):
        """
        Classify the dofs by the the entities.
        """
        p = self.p
        mesh = self.mesh
        TD = mesh.top_dimension()
        NS = TD*(TD+1)//2
        multiindex = bm.multi_index_matrix(self.p, TD)

        boundary_dofs = [[] for i in range(TD+1)]
        internal_dofs = [[] for i in range(TD+1)]
        for i in range(TD+1):
            fs = self.subsimplex[i] 
            fds = self.dual_subsimplex[i] 
            for j in range(len(fs)):
                flag0 = bm.all(multiindex[:, fs[j]] != 0, axis=-1)
                flag1 = bm.all(multiindex[:, fds[j]] == 0, axis=-1)
                flag =  flag0 & flag1 
                idx = bm.where(flag)[0]
                
                N_c = NS-i*(i+1)//2 # 连续标架的个数

                dof_cotinuous = [(alpha, num) for alpha in multiindex[idx] for num in range(N_c)]
                dof_discontinuous = [(alpha, num) for alpha in multiindex[idx] for num in range(N_c, NS)]

                if len(dof_cotinuous) > 0:
                    boundary_dofs[i].append(TensorDofsOnSubsimplex(dof_cotinuous, fs[j]))
                if len(dof_discontinuous) > 0:
                    internal_dofs[i].append(TensorDofsOnSubsimplex(dof_discontinuous, fs[j]))
        return boundary_dofs, internal_dofs 

    def get_boundary_dof_from_dim(self, d):
        """
        Get the dofs of the entities of dimension d.
        """
        return self.boundary_dofs[d]

    def get_internal_dof_from_dim(self, d):
        """
        Get the dofs of the entities of dimension d.
        """
        return self.internal_dofs[d]

class HuZhangFEDof2d():
    """ 
    @brief: The class of HuZhang finite element space dofs.
    @note: Only support the simplicial mesh, the order of  
            local edge, face of the mesh is the same as the order of subsimplex.
    """
    def __init__(self, mesh: Mesh, p: int):
        self.mesh = mesh
        self.p = p
        self.ftype = mesh.ftype
        self.itype = mesh.itype
        self.device = mesh.device

        self.cell_dofs = HuZhangFECellDof2d(mesh, p)

    def number_of_local_dofs(self) -> int:
        """
        Get the number of local dofs on cell 
        """
        p = self.p
        TD = self.mesh.top_dimension()
        NS = TD*(TD+1)//2 # 对称矩阵的自由度个数
        return NS*(p+1)*(p+2)//2 

    def number_of_internal_local_dofs(self, doftype : str='cell') -> int:
        """
        Get the number of internal local dofs of the finite element space.
        """
        p = self.p
        TD = self.mesh.top_dimension()
        NS = TD*(TD+1)//2
        ldof = self.number_of_local_dofs()
        if doftype == 'cell':
            return ldof - NS*3 - 2*(p-1)*3
        elif doftype == 'face' or doftype == 'edge':
            return 2*(p-1) 
        elif doftype == 'node':
            return NS
        else:
            raise ValueError("Unknown doftype: {}".format(doftype))

    def number_of_global_dofs(self) -> int:
        """
        Get the number of global dofs of the finite element space.
        """
        mesh = self.mesh
        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()
        NN = mesh.number_of_nodes()

        cldof = self.number_of_internal_local_dofs('cell')
        eldof = self.number_of_internal_local_dofs('edge')
        nldof = self.number_of_internal_local_dofs('node')
        return NC*cldof + NE*eldof + NN*nldof

    def node_to_internal_dof(self) -> TensorLike:
        """
        Get the index array of the dofs defined on the nodes of the mesh.
        """
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        nldof = self.number_of_internal_local_dofs('node')

        node2dof = bm.arange(NN*nldof, dtype=self.itype, device=self.device)
        return node2dof.reshape(NN, nldof)

    node_to_dof = node_to_internal_dof

    def edge_to_internal_dof(self) -> TensorLike:
        """
        Get the index array of the dofs defined on the edges of the mesh.
        """
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        nldof = self.number_of_internal_local_dofs('node')
        eldof = self.number_of_internal_local_dofs('edge')

        N = NN*nldof
        edge2dof = bm.arange(N, N+NE*eldof, dtype=self.itype, device=self.device)
        return edge2dof.reshape(NE, eldof)

    def edge_to_dof(self, index: Index=_S) -> TensorLike:
        pass

    def cell_to_internal_dof(self) -> TensorLike:
        """
        Get the index array of the dofs defined on the cells of the mesh.
        """
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        nldof = self.number_of_internal_local_dofs('node')
        eldof = self.number_of_internal_local_dofs('edge')
        cldof = self.number_of_internal_local_dofs('cell')

        N = NN*nldof + NE*eldof
        cell2dof = bm.arange(N, N+NC*cldof, dtype=self.itype, device=self.device)
        return cell2dof.reshape(NC, cldof)

    def cell_to_dof(self, index: Index=_S) -> TensorLike:
        """
        Get the cell to dof map of the finite element space.
        """
        p = self.p
        mesh = self.mesh
        ldof = self.number_of_local_dofs()

        NC = mesh.number_of_cells()
        cell = mesh.entity('cell')
        edge = mesh.entity('edge')
        c2e  = mesh.cell_to_edge()

        ndofs = self.cell_dofs.get_boundary_dof_from_dim(0)
        edofs = self.cell_dofs.get_boundary_dof_from_dim(1)

        node2idof = self.node_to_internal_dof()
        edge2idof = self.edge_to_internal_dof()
        cell2idof = self.cell_to_internal_dof()

        c2d = bm.zeros((NC, ldof), dtype=self.itype, device=self.device)
        idx = 0 # 统计自由度的个数

        # 顶点自由度
        for v, dof in enumerate(ndofs):
            n = len(dof)
            c2d[:, idx:idx+n] = node2idof[cell[:, v]]
            idx += n

        # 边自由度
        inverse_perm = [1, 0]
        for e, dof in enumerate(edofs):
            n = len(dof)

            le = bm.sort(mesh.localEdge[e])
            flag = cell[:, le[0]] != edge[c2e[:, e], 0]

            c2d[:, idx:idx+n] = edge2idof[c2e[:, e]]

            inverse_dofidx = dof.permute_to_order(inverse_perm)
            c2d[flag, idx:idx+n] = edge2idof[c2e[flag, e]][:, inverse_dofidx]
            idx += n

        # 单元自由度
        c2d[:, idx:] = cell2idof
        return c2d

    def is_boundary_dof(self, threshold=None, method=None) -> TensorLike:
        """
        Get the bool array of the boundary dofs.
        """
        pass

class HuZhangFESpace2d(FunctionSpace):
    def __init__(self, mesh, p: int=1, ctype='C'):
        self.mesh = mesh
        self.p = p

        self.dof = HuZhangFEDof2d(mesh, p)

        self.ftype = mesh.ftype
        self.itype = mesh.itype

        self.device = mesh.device
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

    def __str__(self):
        return "HuZhangFESpace on {} with p={}".format(self.mesh, self.p)

    ## 自由度接口
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
        return self.dof.is_boundary_dof(threshold, method=method)

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    def project(self, u: Union[Callable[..., TensorLike], TensorLike],) -> TensorLike:
        pass

    def interpolate(self, u: Union[Callable[..., TensorLike], TensorLike],) -> TensorLike:
        pass

    def boundary_interpolate(self,
            gd: Union[Callable, int, float, TensorLike],
            uh: Optional[TensorLike] = None,
            *, threshold: Optional[Threshold]=None, method=None) -> TensorLike:
        #return self.function(uh), isDDof
        pass

    set_dirichlet_bc = boundary_interpolate

    def dof_frame(self) -> TensorLike:
        mesh = self.mesh

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        nframe = bm.zeros((NN, 2, 2), dtype=mesh.ftype)
        eframe = bm.zeros((NE, 2, 2), dtype=mesh.ftype)
        cframe = bm.zeros((NC, 2, 2), dtype=mesh.ftype)

        node = mesh.entity('node')
        edge = mesh.entity('edge')
        cell = mesh.entity('cell')

        nframe[:, 0] = bm.array([[1, 0]], dtype=mesh.ftype) 
        nframe[:, 1] = bm.array([[0, 1]], dtype=mesh.ftype)
        cframe[:, 0] = bm.array([[1, 0]], dtype=mesh.ftype)
        cframe[:, 1] = bm.array([[0, 1]], dtype=mesh.ftype)

        eframe[:, 0] = mesh.edge_unit_normal()
        eframe[:, 1] = mesh.edge_unit_tangent()
        return nframe, eframe, cframe

    def dof_frame_of_S(self):
        mesh = self.mesh

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        nframe, eframe, cframe = self.dof_frame()
        multiindex = bm.multi_index_matrix(2, 1)
        idx, num = symmetry_index(2, 2)

        nsframe = bm.zeros((NN, 3, 3), dtype=self.ftype)
        for i, alpha in enumerate(multiindex): 
            nsframe[:, i] = symmetry_span_array(nframe, alpha).reshape(NN, -1)[:, idx]

        esframe = bm.zeros((NE, 3, 3), dtype=self.ftype)
        for i, alpha in enumerate(multiindex): 
            esframe[:, i] = symmetry_span_array(eframe, alpha).reshape(NE, -1)[:, idx]

        csframe = bm.zeros((NC, 3, 3), dtype=self.ftype)
        for i, alpha in enumerate(multiindex): 
            csframe[:, i] = symmetry_span_array(cframe, alpha).reshape(NC, -1)[:, idx]
        return nsframe, esframe, csframe

    basis_frame = dof_frame
    basis_frame_of_S = dof_frame_of_S

    def basis(self, bc: TensorLike, index: Index=_S):
        p = self.p
        mesh = self.mesh
        dof = self.dof

        ldof = dof.number_of_local_dofs()

        ndofs = dof.cell_dofs.get_boundary_dof_from_dim(0)
        edofs = dof.cell_dofs.get_boundary_dof_from_dim(1)

        iedofs = dof.cell_dofs.get_internal_dof_from_dim(1)
        icdofs = dof.cell_dofs.get_internal_dof_from_dim(2)


        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        cell = mesh.entity('cell')
        c2e = mesh.cell_to_edge()

        nsframe, esframe, csframe = self.basis_frame_of_S()

        phi_s = self.mesh.shape_function(bc, self.p, index=index) # (NC, NQ, ldof)

        NQ = bc.shape[0]
        phi = bm.zeros((NC, NQ, ldof, 3), dtype=self.ftype)

        # 顶点基函数
        idx = 0
        for v, vdof in enumerate(ndofs):
            N = len(vdof)
            scalar_phi_idx = multiindex_to_number(vdof.dof_scalar)
            scalar_part = phi_s[None, :, scalar_phi_idx, None]
            tensor_part = nsframe[cell[:, v]][:, None, vdof.dof_tensor, :]
            phi[..., idx:idx+N, :] = scalar_part * tensor_part
            idx += N

        # 边基函数
        for e, edof in enumerate(edofs):
            N = len(edof)
            scalar_phi_idx = multiindex_to_number(edof.dof_scalar)
            scalar_part = phi_s[None, :, scalar_phi_idx, None]
            tensor_part = esframe[c2e[:, e]][:, None, edof.dof_tensor, :]
            phi[..., idx:idx+N, :] = scalar_part * tensor_part
            idx += N

        # 单元基函数
        for e, edof in enumerate(iedofs):
            N = len(edof)
            scalar_phi_idx = multiindex_to_number(edof.dof_scalar)
            scalar_part = phi_s[None, :, scalar_phi_idx, None]
            tensor_part = esframe[c2e[:, e]][:, None, edof.dof_tensor, :]
            phi[..., idx:idx+N, :] = scalar_part * tensor_part
            idx += N
        scalar_phi_idx = multiindex_to_number(icdofs[0].dof_scalar)
        scalar_part = phi_s[None, :, scalar_phi_idx, None]
        tensor_part = csframe[:, None, icdofs[0].dof_tensor, :]
        phi[..., idx:, :] = scalar_part * tensor_part
        return phi

    def div_basis(self, bc: TensorLike): 
        p = self.p
        mesh = self.mesh
        dof = self.dof

        ldof = dof.number_of_local_dofs()

        ndofs = dof.cell_dofs.get_boundary_dof_from_dim(0)
        edofs = dof.cell_dofs.get_boundary_dof_from_dim(1)

        iedofs = dof.cell_dofs.get_internal_dof_from_dim(1)
        icdofs = dof.cell_dofs.get_internal_dof_from_dim(2)

        cell = mesh.entity('cell')
        c2e  = mesh.cell_to_edge()

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        nsframe, esframe, csframe = self.basis_frame_of_S()

        gphi_s = self.mesh.grad_shape_function(bc, self.p) # (NC, ldof, GD)

        NQ = bc.shape[0]
        dphi = bm.zeros((NC, NQ, ldof, 2), dtype=self.ftype)

        # 顶点基函数
        idx = 0
        for v, vdof in enumerate(ndofs):
            N = len(vdof)
            scalar_phi_idx = multiindex_to_number(vdof.dof_scalar)
            grad_scalar = gphi_s[..., scalar_phi_idx, :] # (NC, NQ, N, 2)
            frame = nsframe[cell[:, v]][:, None, vdof.dof_tensor] # (NC, 1, N, 3)
            dphi[..., idx:idx+N, 0] = bm.sum(grad_scalar * frame[..., :2], axis=-1)
            dphi[..., idx:idx+N, 1] = bm.sum(grad_scalar * frame[..., 1:], axis=-1)
            idx += N

        # 边基函数
        for e, edof in enumerate(edofs):
            N = len(edof)
            scalar_phi_idx = multiindex_to_number(edof.dof_scalar)
            grad_scalar = gphi_s[..., scalar_phi_idx, :]
            frame = esframe[c2e[:, e]][:, None, edof.dof_tensor]
            dphi[..., idx:idx+N, 0] = bm.sum(grad_scalar * frame[..., :2], axis=-1)
            dphi[..., idx:idx+N, 1] = bm.sum(grad_scalar * frame[..., 1:], axis=-1)
            idx += N

        # 单元基函数
        for e, edof in enumerate(iedofs):
            N = len(edof)
            scalar_phi_idx = multiindex_to_number(edof.dof_scalar)
            grad_scalar = gphi_s[..., scalar_phi_idx, :]
            frame = esframe[c2e[:, e]][:, None, edof.dof_tensor]
            dphi[..., idx:idx+N, 0] = bm.sum(grad_scalar * frame[..., :2], axis=-1)
            dphi[..., idx:idx+N, 1] = bm.sum(grad_scalar * frame[..., 1:], axis=-1)
            idx += N

        scalar_phi_idx = multiindex_to_number(icdofs[0].dof_scalar)
        grad_scalar = gphi_s[..., scalar_phi_idx, :]
        frame = csframe[:, None, icdofs[0].dof_tensor]
        dphi[..., idx:, 0] = bm.sum(grad_scalar * frame[..., :2], axis=-1)
        dphi[..., idx:, 1] = bm.sum(grad_scalar * frame[..., 1:], axis=-1)
        return dphi

    def hess_basis(self, bc: TensorLike, index: Index=_S, variable='x'):
        return self.mesh.hess_shape_function(bc, self.p, index=index, variables=variable)

    @barycentric
    def value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike: 
        if isinstance(bc, tuple):
            TD = len(bc)
        else :
            TD = bc.shape[-1] - 1
        phi = self.basis(bc, index=index)
        e2dof = self.dof.cell_to_dof()
        val = bm.einsum('cqld, ...cl -> ...cqd', phi, uh[..., e2dof])
        return val

    @barycentric
    def div_value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike:
        if isinstance(bc, tuple):
            TD = len(bc)
        else :
            TD = bc.shape[-1] - 1
        gphi = self.grad_basis(bc, index=index)
        e2dof = self.dof.entity_to_dof(TD, index=index)
        val = bm.einsum('cilm, cl -> cim', gphi, uh[e2dof])
        return val
    
