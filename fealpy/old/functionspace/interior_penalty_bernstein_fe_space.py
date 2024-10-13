
import numpy as np
from itertools import product

from .fem_dofs import TriangleMeshCFEDof
from .bernstein_fe_space import BernsteinFESpace 

class InteriorPenaltyBernsteinDof2d(TriangleMeshCFEDof):
    """
    @brief 内罚 Bernstein 元的自由度，继承于连续 Bernstein 元的自由度，添
           加了一条边上周围两个单元上的自由度
    """
    def __init__(self, mesh, p):
        super(InteriorPenaltyBernsteinDof2d, self).__init__(mesh, p)
        self.iedge2celldof = self.inner_edge_to_cell_dof()
        self.bedge2celldof = self.boundary_edge_to_cell_dof()

    def inner_edge_to_cell_dof(self):
        """
        @brief 一条边周围两个单元的自由度的编号
        """

        c2d = self.cell2dof
        e2d = self.edge_to_dof()
        e2c = self.mesh.ds.edge2cell

        isInnerEdge = ~self.mesh.ds.boundary_edge_flag() 

        NIE   = isInnerEdge.sum()
        ie2c = e2c[isInnerEdge]

        edof = self.number_of_local_dofs('edge')
        cdof = self.number_of_local_dofs('cell')
        ldof = 2*cdof - edof 

        # 左边单元的自由度
        ie2cd0 = np.zeros([NIE, cdof-edof], dtype=self.mesh.itype) 
        for i in range(3):
            edgeidx = ie2c[:, 2]==i
            dofidx  = self.multiIndex[:, i] != 0
            ie2cd0[edgeidx] = c2d[ie2c[edgeidx, 0]][:, dofidx]

        # 右边单元的自由度
        ie2cd1 = np.zeros([NIE, cdof-edof], dtype=self.mesh.itype) 
        for i in range(3):
            edgeidx = ie2c[:, 3]==i
            dofidx  = self.multiIndex[:, i] != 0
            ie2cd1[edgeidx] = c2d[ie2c[edgeidx, 1]][:, dofidx]

        # 边上的自由度
        ie2cd2 = e2d[isInnerEdge]
        ie2cd  = np.concatenate([ie2cd0, ie2cd1, ie2cd2], axis=1)
        return ie2cd

    def boundary_edge_to_cell_dof(self):
        mesh = self.mesh
        e2c = mesh.ds.edge2cell

        isBdEdge = mesh.ds.boundary_edge_flag()
        be2c = e2c[isBdEdge]

        c2d = self.cell2dof
        return c2d[be2c[:, 0]]

class InteriorPenaltyBernsteinFESpace2d(BernsteinFESpace):
    """
    @brief 内罚 Bernstein 元，继承于 Bernstein 元，添加了 Bernstein
           基函数在边上的罚项计算
    """
    def __init__(self, mesh, p=2):
        super(InteriorPenaltyBernsteinFESpace2d, self).__init__(mesh, p, 'C')
        self.dof = InteriorPenaltyBernsteinDof2d(mesh, p)
        self.ftype = np.float64

    def grad_normal_jump_basis(self, bcs, m=1):
        """
        @brief 法向导数跳量计算
        @return (NQ, NIE, ldof)
        """
        mesh = self.mesh
        e2c  = mesh.ds.edge2cell
        edof = self.dof.number_of_local_dofs('edge')
        cdof = self.dof.number_of_local_dofs('cell')
        ldof = 2*cdof - edof 

        # 内部边
        isInnerEdge = ~mesh.ds.boundary_edge_flag()
        NIE  = isInnerEdge.sum()
        ie2c = e2c[isInnerEdge]
        en   = mesh.edge_unit_normal()[isInnerEdge]

        # 扩充重心坐标 
        shape = bcs.shape[:-1]
        bcss = [np.insert(bcs, i, 0, axis=-1) for i in range(3)]
        bcss[1] = bcss[1][..., [2, 1, 0]]

        edof2lcdof = [slice(None), slice(None, None, -1), slice(None)]
        edof2rcdof = [slice(None, None, -1), slice(None), slice(None, None, -1)]

        rval0  = np.zeros(shape+(NIE, cdof-edof), dtype=self.mesh.ftype)
        rval1  = np.zeros(shape+(NIE, cdof-edof), dtype=self.mesh.ftype)
        rval2  = np.zeros(shape+(NIE,      edof), dtype=self.mesh.ftype)

        # 左边单元的基函数的法向导数跳量
        for i in range(3):
            bcsi    = bcss[i] 
            edgeidx = ie2c[:, 2]==i
            dofidx0 = np.where(self.dof.multiIndex[:, i] != 0)[0]
            dofidx1 = np.where(self.dof.multiIndex[:, i] == 0)[0][edof2lcdof[i]]

            gval = self.grad_basis(bcsi, index=ie2c[edgeidx, 0])
            val  = np.einsum('qedi, ei->qed', gval, en[edgeidx]) # (NQ, NIEi, cdof)

            rval0[..., edgeidx, :] = val[..., dofidx0]
            rval2[..., edgeidx, :] += val[..., dofidx1]

        bcss = [np.insert(bcs[..., ::-1], i, 0, axis=-1) for i in range(3)]
        bcss[1] = bcss[1][..., [2, 1, 0]]

        # 右边单元的基函数的法向导数跳量
        for i in range(3):
            bcsi    = bcss[i] 
            edgeidx = ie2c[:, 3]==i
            dofidx0 = np.where(self.dof.multiIndex[:, i] != 0)[0]
            dofidx1 = np.where(self.dof.multiIndex[:, i] == 0)[0][edof2rcdof[i]]

            gval = self.grad_basis(bcsi, index=ie2c[edgeidx, 1])
            val  = np.einsum('qedi, ei->qed', gval, -en[edgeidx]) # (NQ, NIEi, cdof)

            rval1[..., edgeidx, :] = val[..., dofidx0]
            rval2[..., edgeidx, :] += val[..., dofidx1]
        rval = np.concatenate([rval0, rval1, rval2], axis=-1)
        return rval

    def grad_normal_2_jump_basis(self, bcs):
        """
        @brief 2 阶法向导数跳量计算
        @return (NQ, NC, ldof)
        """
        mesh = self.mesh
        e2c  = mesh.ds.edge2cell
        edof = self.dof.number_of_local_dofs('edge')
        cdof = self.dof.number_of_local_dofs('cell')
        ldof = 2*cdof - edof 

        # 内部边
        isInnerEdge = ~mesh.ds.boundary_edge_flag()
        NIE  = isInnerEdge.sum()
        ie2c = e2c[isInnerEdge]
        en   = mesh.edge_unit_normal()[isInnerEdge]

        # 扩充重心坐标 
        shape = bcs.shape[:-1]
        bcss = [np.insert(bcs, i, 0, axis=-1) for i in range(3)]
        bcss[1] = bcss[1][..., [2, 1, 0]]

        edof2lcdof = [slice(None), slice(None, None, -1), slice(None)]

        rval0  = np.zeros(shape+(NIE, cdof-edof), dtype=self.mesh.ftype)
        rval1  = np.zeros(shape+(NIE, cdof-edof), dtype=self.mesh.ftype)
        rval2  = np.zeros(shape+(NIE,      edof), dtype=self.mesh.ftype)

        # 左边单元
        for i in range(3):
            bcsi    = bcss[i] 
            edgeidx = ie2c[:, 2]==i
            dofidx0 = np.where(self.dof.multiIndex[:, i] != 0)[0]
            dofidx1 = np.where(self.dof.multiIndex[:, i] == 0)[0][edof2lcdof[i]]

            hval = self.hess_basis(bcsi, index=ie2c[edgeidx, 0])
            val  = np.einsum('qedij, ei, ej->qed', hval, en[edgeidx],
                    en[edgeidx])/2# (NQ, NIEi, cdof)

            indices = (Ellipsis, edgeidx, slice(None))
            rval0[indices] = val[..., dofidx0]
            rval2[indices] += val[..., dofidx1]

        bcss = [np.insert(bcs[..., ::-1], i, 0, axis=-1) for i in range(3)]
        bcss[1] = bcss[1][..., [2, 1, 0]]
        edof2rcdof = [slice(None, None, -1), slice(None), slice(None, None, -1)]

        # 右边单元
        for i in range(3):
            bcsi    = bcss[i] 
            edgeidx = ie2c[:, 3]==i
            dofidx0 = np.where(self.dof.multiIndex[:, i] != 0)[0]
            dofidx1 = np.where(self.dof.multiIndex[:, i] == 0)[0][edof2rcdof[i]]

            hval = self.hess_basis(bcsi, index=ie2c[edgeidx, 1])
            val  = np.einsum('qedij, ei, ej->qed', hval, en[edgeidx],
                    en[edgeidx])/2# (NQ, NIEi, cdof)

            indices = (Ellipsis, edgeidx, slice(None))
            rval1[indices] = val[..., dofidx0]
            rval2[indices] += val[..., dofidx1]
        rval = np.concatenate([rval0, rval1, rval2], axis=-1)
        return rval

    def boubdary_edge_grad_normal_jump_basis(self, bcs, m=1):
        """
        @brief 法向导数跳量计算
        @return (NQ, NIE, ldof)
        """
        mesh = self.mesh
        e2c  = mesh.ds.edge2cell
        cdof = self.dof.number_of_local_dofs('cell')

        # 边界边
        isBdEdge = mesh.ds.boundary_edge_flag()
        NBE  = isBdEdge.sum()
        be2c = e2c[isBdEdge]
        en   = mesh.edge_unit_normal()[isBdEdge]

        # 扩充重心坐标 
        shape = bcs.shape[:-1]
        bcss = [np.insert(bcs, i, 0, axis=-1) for i in range(3)]
        bcss[1] = bcss[1][..., [2, 1, 0]]

        rval = np.zeros(shape+(NBE, cdof), dtype=self.mesh.ftype)

        # 左边单元的基函数的法向导数跳量
        for i in range(3):
            bcsi    = bcss[i] 
            edgeidx = be2c[:, 2]==i

            gval = self.grad_basis(bcsi, index=be2c[edgeidx, 0])
            rval[..., edgeidx, :] = np.einsum('qedi, ei->qed', gval, en[edgeidx]) # (NQ, NIEi, cdof)
        return rval

    def boubdary_edge_grad_normal_2_jump_basis(self, bcs, m=1):
        """
        @brief 法向导数跳量计算
        @return (NQ, NIE, ldof)
        """
        mesh = self.mesh
        e2c  = mesh.ds.edge2cell
        cdof = self.dof.number_of_local_dofs('cell')

        # 边界边
        isBdEdge = mesh.ds.boundary_edge_flag()
        NBE  = isBdEdge.sum()
        be2c = e2c[isBdEdge]
        en   = mesh.edge_unit_normal()[isBdEdge]

        # 扩充重心坐标 
        shape = bcs.shape[:-1]
        bcss = [np.insert(bcs, i, 0, axis=-1) for i in range(3)]
        bcss[1] = bcss[1][..., [2, 1, 0]]

        rval = np.zeros(shape+(NBE, cdof), dtype=self.mesh.ftype)

        # 左边单元的基函数的法向导数跳量
        for i in range(3):
            bcsi    = bcss[i] 
            edgeidx = be2c[:, 2]==i

            hval = self.hess_basis(bcsi, index=be2c[edgeidx, 0])
            rval[..., edgeidx, :] = np.einsum('qedij, ei, ej->qed', hval, en[edgeidx], en[edgeidx])# (NQ, NIEi, cdof)
        return rval
        





