from typing import Optional, TypeVar, Union, Generic, Callable
from ..typing import TensorLike, Index, _S, Threshold

from ..backend import TensorLike
from ..backend import backend_manager as bm
from ..mesh.mesh_base import Mesh
from .lagrange_fe_space import LagrangeFESpace
from .bernstein_fe_space import BernsteinFESpace
from .dofs import LinearMeshCFEDof, LinearMeshDFEDof

import numpy as np


_MT = TypeVar('_MT', bound=Mesh)

class InteriorPenaltyDof2d(LinearMeshCFEDof):
    """
    2D 的内部罚函数拉格朗日有限元空间的自由度
    """
    def __init__(self, mesh, p):
        super(InteriorPenaltyDof2d, self).__init__(mesh, p)
        self.iedge2celldof = self.inner_edge_to_cell_dof()
        self.bedge2celldof = self.boundary_edge_to_cell_dof()

    def inner_edge_to_cell_dof(self):
        """
        @brief 一条边周围两个单元的自由度的编号
        """

        c2d = self.cell_to_dof()
        e2d = self.edge_to_dof()
        e2c = self.mesh.edge_to_cell()

        isInnerEdge = ~self.mesh.boundary_edge_flag() 

        NIE   = isInnerEdge.sum()
        ie2c = e2c[isInnerEdge]

        edof = self.number_of_local_dofs('edge')
        cdof = self.number_of_local_dofs('cell')
        ldof = 2*cdof - edof 

        # 左边单元的自由度
        ie2cd0 = bm.zeros([NIE, cdof-edof], dtype=self.mesh.itype) 
        for i in range(3):
            edgeidx = ie2c[:, 2]==i
            dofidx  = self.multiIndex[:, i] != 0
            ie2cd0 = bm.set_at(ie2cd0, edgeidx, c2d[ie2c[edgeidx, 0]][:, dofidx])
        # 右边单元的自由度
        ie2cd1 = bm.zeros([NIE, cdof-edof], dtype=self.mesh.itype) 
        for i in range(3):
            edgeidx = ie2c[:, 3]==i
            dofidx  = self.multiIndex[:, i] != 0
            ie2cd1 = bm.set_at(ie2cd1, edgeidx, c2d[ie2c[edgeidx, 1]][:, dofidx])
        # 边上的自由度
        ie2cd2 = e2d[isInnerEdge]
        ie2cd  = bm.concatenate([ie2cd0, ie2cd1, ie2cd2], axis=1)
        return ie2cd

    def boundary_edge_to_cell_dof(self):
        mesh = self.mesh
        e2c = mesh.edge_to_cell()

        isBdEdge = mesh.boundary_edge_flag()
        be2c = e2c[isBdEdge]

        c2d = self.cell_to_dof()
        return c2d[be2c[:, 0]]


class InteriorPenaltyFESpace2d:
    def __init__(self, mesh, p=2, space=None, ctype='C'):
        self.dof = InteriorPenaltyDof2d(mesh, p)
        if space is None or space == 'Lagrange':
            self.base_space = LagrangeFESpace(mesh, p, ctype=ctype)
        elif space == 'Bernstein':
            self.base_space = BernsteinFESpace(mesh, p, ctype=ctype)
        else:
            raise ValueError("Space must be either 'Lagrange' or 'Bernstein'")


    def __getattr__(self, name):
        # 当访问不存在的属性时，尝试从base_space获取
        return getattr(self.base_space, name)
        

    def grad_normal_jump_basis(self, bcs, m=1):
        """
        @brief 法向导数跳量计算
        @return (NQ, NIE, ldof)
        """
        mesh = self.mesh
        e2c  = mesh.edge_to_cell()
        edof = self.number_of_local_dofs('edge')
        cdof = self.number_of_local_dofs('cell')
        ldof = 2*cdof - edof 

        # 内部边
        isInnerEdge = ~mesh.boundary_edge_flag()
        NIE  = isInnerEdge.sum()
        ie2c = e2c[isInnerEdge]
        en   = mesh.edge_unit_normal()[isInnerEdge]

        # 扩充重心坐标 
        shape = bcs.shape[:-1]
        bcss = [bm.insert(bcs, i, 0, axis=-1) for i in range(3)]
        bcss[1] = bcss[1][..., [2, 1, 0]]

        slice_operations = {
            0: lambda x: x,  # slice(None) -> 保持原样
            1: lambda x: bm.flip(x)  # slice(None, None, -1) -> 反转
        }

        # 初始化 edof2lcdof 和 edof2rcdof 以匹配 slice_operations
        edof2lcdof = [0, 1, 0]  # 索引对应 slice(None), slice(None, None, -1), slice(None)
        edof2rcdof = [1, 0, 1]  # 索引对应 slice(None, None, -1), slice(None), slice(None, None, -1)

        #edof2lcdof = [slice(None), slice(None, None, -1), slice(None)]
        #edof2rcdof = [slice(None, None, -1), slice(None), slice(None, None, -1)]

        rval0  = bm.zeros(shape+(NIE, cdof-edof), dtype=self.mesh.ftype)
        rval1  = bm.zeros(shape+(NIE, cdof-edof), dtype=self.mesh.ftype)
        rval2  = bm.zeros(shape+(NIE,      edof), dtype=self.mesh.ftype)

        # 左边单元的基函数的法向导数跳量
        for i in range(3):
            bcsi    = bcss[i] 
            
            edgeidx = ie2c[:, 2]==i
            

            # 应用操作映射来获取正确的索引
            zero_indices = bm.where(self.dof.multiIndex[:, i] == 0)[0]
            dofidx1 = slice_operations[edof2lcdof[i]](zero_indices)
            
            dofidx0 = bm.where(self.dof.multiIndex[:, i] != 0)[0]
            #dofidx1 = bm.where(self.dof.multiIndex[:, i] == 0)[0][edof2lcdof[i]]

            gval = self.grad_basis(bcsi, index=ie2c[edgeidx, 0], variable='x')
            val  = bm.einsum('eqdi, ei->qed', gval, en[edgeidx]) # (NQ, NIEi, cdof)

            indices = (Ellipsis, edgeidx, slice(None))
            rval0 = bm.set_at(rval0, indices, val[..., dofidx0])
            rval2 = bm.add_at(rval2, indices, val[..., dofidx1])

        bcss = [bm.insert(bm.flip(bcs, axis=-1), i, 0, axis=-1) for i in range(3)]
        bcss[1] = bcss[1][..., [2, 1, 0]]
        # 右边单元的基函数的法向导数跳量
        for i in range(3):
            bcsi    = bcss[i] 
            edgeidx = ie2c[:, 3]==i
            dofidx0 = bm.where(self.dof.multiIndex[:, i] != 0)[0]

            # 应用操作映射来获取正确的索引
            zero_indices = bm.where(self.dof.multiIndex[:, i] == 0)[0]
            dofidx1 = slice_operations[edof2rcdof[i]](zero_indices)
            #dofidx1 = bm.where(self.dof.multiIndex[:, i] == 0)[0][edof2rcdof[i]]
            
            gval = self.grad_basis(bcsi, index=ie2c[edgeidx, 1], variable='x')
            val  = bm.einsum('eqdi, ei->qed', gval, -en[edgeidx]) # (NQ, NIEi, cdof)

            indices = (Ellipsis, edgeidx, slice(None))
            rval1 = bm.set_at(rval1, indices, val[..., dofidx0])
            rval2 = bm.add_at(rval2, indices, val[..., dofidx1])
        rval = bm.concatenate([rval0, rval1, rval2], axis=-1)
        return rval

    def grad_grad_normal_jump_basis(self, bcs):
        """
        @brief 2 阶法向导数跳量计算
        @return (NQ, NC, ldof)
        """
        mesh = self.mesh
        e2c  = mesh.edge_to_cell()
        edof = self.number_of_local_dofs('edge')
        cdof = self.number_of_local_dofs('cell')
        ldof = 2*cdof - edof 

        # 内部边
        isInnerEdge = ~mesh.boundary_edge_flag()
        NIE  = isInnerEdge.sum()
        ie2c = e2c[isInnerEdge]
        en   = mesh.edge_unit_normal()[isInnerEdge]

        # 扩充重心坐标 
        shape = bcs.shape[:-1]
        bcss = [bm.insert(bcs, i, 0, axis=-1) for i in range(3)]
        bcss[1] = bcss[1][..., [2, 1, 0]]

        slice_operations = {
            0: lambda x: x,  # slice(None) -> 保持原样
            1: lambda x: bm.flip(x)  # slice(None, None, -1) -> 反转
        }

        # 初始化 edof2lcdof 和 edof2rcdof 以匹配 slice_operations
        edof2lcdof = [0, 1, 0]  # 索引 0, 1 对应 slice(None), slice(None, None, -1), slice(None)
        edof2rcdof = [1, 0, 1]  # 索引 0, 1 对应 slice(None, None, -1), slice(None), slice(None, None, -1)

        #edof2lcdof = [slice(None), slice(None, None, -1), slice(None)]
        #edof2rcdof = [slice(None, None, -1), slice(None), slice(None, None, -1)]

        rval0  = bm.zeros(shape+(NIE, cdof-edof), dtype=self.mesh.ftype)
        rval1  = bm.zeros(shape+(NIE, cdof-edof), dtype=self.mesh.ftype)
        rval2  = bm.zeros(shape+(NIE,      edof), dtype=self.mesh.ftype)

        # 左边单元
        for i in range(3):
            bcsi    = bcss[i] 
            edgeidx = ie2c[:, 2]==i
            dofidx0 = bm.where(self.dof.multiIndex[:, i] != 0)[0]
            zero_indices = bm.where(self.dof.multiIndex[:, i] == 0)[0]
            dofidx1 = slice_operations[edof2lcdof[i]](zero_indices)
            #dofidx1 = bm.where(self.dof.multiIndex[:, i] == 0)[0][edof2lcdof[i]]
            
            idx = ie2c[edgeidx, 0]
            if len(idx) == 0:
                val = bm.zeros((bcsi.shape[0], len(idx), cdof), dtype=self.mesh.ftype)
            else:
                hval = self.hess_basis(bcsi, index=idx, variable='x')
                val  = bm.einsum('eqdij, ei, ej->qed', hval, en[edgeidx],
                        en[edgeidx])/2.0# (NQ, NIEi, cdof)

            indices = (Ellipsis, edgeidx, slice(None))
            rval0 = bm.set_at(rval0, indices, val[..., dofidx0])
            rval2 = bm.add_at(rval2, indices, val[..., dofidx1])

        bcss = [bm.insert(bm.flip(bcs, axis=-1), i, 0, axis=-1) for i in range(3)]
        bcss[1] = bcss[1][..., [2, 1, 0]]
        # 右边单元
        for i in range(3):
            bcsi    = bcss[i] 
            edgeidx = ie2c[:, 3]==i
            dofidx0 = bm.where(self.dof.multiIndex[:, i] != 0)[0]
            zero_indices = bm.where(self.dof.multiIndex[:, i] == 0)[0]
            dofidx1 = slice_operations[edof2rcdof[i]](zero_indices)
            #dofidx1 = bm.where(self.dof.multiIndex[:, i] == 0)[0][edof2rcdof[i]]
            
            idx = ie2c[edgeidx, 1]
            if len(idx) == 0:
                val = bm.zeros((bcsi.shape[0], len(idx), cdof), dtype=self.mesh.ftype)
            else:
                hval = self.hess_basis(bcsi, index=idx, variable='x')
                val  = bm.einsum('eqdij, ei, ej->qed', hval, en[edgeidx],
                        en[edgeidx])/2.0# (NQ, NIEi, cdof)

            indices = (Ellipsis, edgeidx, slice(None))
            rval1 = bm.set_at(rval1, indices, val[..., dofidx0])
            rval2 = bm.add_at(rval2, indices, val[..., dofidx1])
        rval = bm.concatenate([rval0, rval1, rval2], axis=-1)
        return rval

    def boundary_edge_grad_normal_jump_basis(self, bcs, m=1):
        """
        @brief 法向导数跳量计算
        @return (NQ, NIE, ldof)
        """
        mesh = self.mesh
        e2c  = mesh.edge_to_cell()
        cdof = self.number_of_local_dofs('cell')

        # 边界边
        isBdEdge = mesh.boundary_edge_flag()
        NBE  = isBdEdge.sum()
        be2c = e2c[isBdEdge]
        en   = mesh.edge_unit_normal()[isBdEdge]

        # 扩充重心坐标 
        shape = bcs.shape[:-1]
        bcss = [bm.insert(bcs, i, 0, axis=-1) for i in range(3)]
        bcss[1] = bcss[1][..., [2, 1, 0]]

        rval = bm.zeros(shape+(NBE, cdof), dtype=self.mesh.ftype)

        # 左边单元的基函数的法向导数跳量
        for i in range(3):
            bcsi    = bcss[i] 
            edgeidx = be2c[:, 2]==i

            gval = self.grad_basis(bcsi, index=be2c[edgeidx, 0], variable='x')
            val  = bm.einsum('eqdi, ei->qed', gval, en[edgeidx]) # (NQ, NIEi, cdof)

            indices = (Ellipsis, edgeidx, slice(None))
            rval = bm.set_at(rval, indices, val)
        return rval

    def boundary_edge_grad_grad_normal_jump_basis(self, bcs, m=1):
        """
        @brief 法向导数跳量计算
        @return (NQ, NIE, ldof)
        """
        mesh = self.mesh
        e2c  = mesh.edge_to_cell()
        cdof = self.number_of_local_dofs('cell')

        # 边界边
        isBdEdge = mesh.boundary_edge_flag()
        NBE  = isBdEdge.sum()
        be2c = e2c[isBdEdge]
        en   = mesh.edge_unit_normal()[isBdEdge]

        # 扩充重心坐标 
        shape = bcs.shape[:-1]
        bcss = [bm.insert(bcs, i, 0, axis=-1) for i in range(3)]
        bcss[1] = bcss[1][..., [2, 1, 0]]

        rval = bm.zeros(shape+(NBE, cdof), dtype=self.mesh.ftype)

        # 左边单元的基函数的法向导数跳量
        for i in range(3):
            bcsi    = bcss[i] 
            edgeidx = be2c[:, 2]==i

            idx = be2c[edgeidx, 0]
            if len(idx) == 0:
                val = bm.zeros((bcsi.shape[0], len(idx), cdof), dtype=self.mesh.ftype)
            else:

                hval = self.hess_basis(bcsi, index=idx, variable='x')
                val  = bm.einsum('eqdij, ei, ej->qed', hval, en[edgeidx], en[edgeidx])# (NQ, NIEi, cdof)

            indices = (Ellipsis, edgeidx, slice(None))
            rval = bm.set_at(rval, indices, val)
        return rval
