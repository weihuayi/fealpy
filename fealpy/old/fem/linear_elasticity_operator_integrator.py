import numpy as np
from typing import Optional, Tuple

from fealpy.old.fem.precomp_data import data


class LinearElasticityOperatorIntegrator:
    def __init__(self, lam, mu, q=None, c=None):
        """
        初始化 LinearElasticityOperatorIntegrator 类

        参数:
        lam (float): 拉梅系数
        mu (float): 剪切模量
        q (Optional[int]): 积分阶次，默认为 None
        """
        self.lam = lam
        self.mu = mu
        self.q = q
        self.c = c
        self.type = "BL0"

    def assembly_cell_matrix(self, space: Tuple, index=np.s_[:],
                             cellmeasure: Optional[np.ndarray] = None,
                             out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        构建线性弹性有限元矩阵

        参数:
        space (Tuple): 有限元空间
        index (Union[np.s_, np.ndarray]): 选定的单元索引，默认为全部单元
        cellmeasure (Optional[np.ndarray]): 对应单元的度量，默认为 None
        out (Optional[np.ndarray]): 输出矩阵，默认为 None

        返回:
        Optional[np.ndarray]: 如果 out 参数为 None，则返回线性弹性有限元矩阵，否则不返回
        """
        ...
        lam = self.lam
        mu = self.mu
        c = self.c
        mesh = space[0].mesh
        ldof = space[0].number_of_local_dofs()
        p = space[0].p  # 空间的多项式阶数
        GD = mesh.geo_dimension()
        q = self.q if self.q is not None else p + 1
        NC = mesh.number_of_cells()

        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        if GD == 2:
            # 每个元组代表一个弹性张量的二阶导数的索引对
            idx = [(0, 0), (0, 1), (1, 1)]
            # 将 idx 中的元组映射到一个整数上
            imap = {(0, 0): 0, (0, 1): 1, (1, 1): 2}
        elif GD == 3:
            idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
            imap = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 1): 3, (1, 2): 4, (2, 2): 5}

        A = []

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        grad = space[0].grad_basis(bcs, index=index)  # (NQ, NC, ldof, GD)
        NQ = len(ws)

        if np.isscalar(cellmeasure):
                cellmeasure = np.full( (NC, ), cellmeasure)
                
        NC = len(cellmeasure)

        if out is None:
            K = np.zeros((NC, GD * ldof, GD * ldof), dtype=np.float64)
        else:
            assert out.shape == (NC, GD * ldof, GD * ldof)
            K = out

        # 对于每一个设定的索引对，利用四边形积分公式和基函数的梯度来计算一个积分项
        if c is None:
            A = [np.einsum('i, ijm, ijn, j->jmn', ws, grad[..., i], grad[..., j], cellmeasure, optimize=True) for i, j
                 in idx]
        else:
            if callable(c):
                if hasattr(c, 'coordtype'):
                    if c.coordtype == 'cartesian':
                        ps = mesh.bc_to_point(bcs, index=index)
                        c = c(ps)
                    elif c.coordtype == 'barycentric':
                        c = c(bcs, index=index)
                else:
                    ps = mesh.bc_to_point(bcs, index=index)
                    c = c(ps)
            if np.isscalar(c):
                A = [c * np.einsum('i, ijm, ijn, j -> jmn', ws, grad[..., i], grad[..., j], cellmeasure,
                                   optimize=True) for i, j in idx]
            elif isinstance(c, np.ndarray):
                if c.shape == (NC,):
                    A = [np.einsum('i, j, ijm, ijn, j -> jmn', ws, c, grad[..., i], grad[..., j], cellmeasure,
                                   optimize=True) for i, j in idx]
                elif c.shape == (NQ, NC):
                    A = [np.einsum('i, ij, ijm, ijn, j -> jmn', ws, c, grad[..., i], grad[..., j], cellmeasure,
                                   optimize=True) for i, j in idx]
                else:
                    raise ValueError(f"coef with shape {c.shape}! Now we just support shape: (NC, ), (NQ, NC)")
            else:
                raise ValueError("coef 不支持该类型")

        D = 0
        for i in range(GD):
            D += mu * A[imap[(i, i)]]
        if space[0].doforder == 'sdofs':  # 标量自由度优先排序 
            for i in range(GD):
                for j in range(i, GD):
                    if i == j:
                        K[:, i * ldof:(i + 1) * ldof, i * ldof:(i + 1) * ldof] += D
                        K[:, i * ldof:(i + 1) * ldof, i * ldof:(i + 1) * ldof] += (mu + lam) * A[imap[(i, i)]]
                    else:
                        K[:, i * ldof:(i + 1) * ldof, j * ldof:(j + 1) * ldof] += lam * A[imap[(i, j)]]
                        K[:, i * ldof:(i + 1) * ldof, j * ldof:(j + 1) * ldof] += mu * A[imap[(i, j)]].transpose(0, 2,
                                                                                                                 1)
                        K[:, j * ldof:(j + 1) * ldof, i * ldof:(i + 1) * ldof] += lam * A[imap[(i, j)]].transpose(0, 2,
                                                                                                                  1)
                        K[:, j * ldof:(j + 1) * ldof, i * ldof:(i + 1) * ldof] += mu * A[imap[(i, j)]]
        elif space[0].doforder == 'vdims':
            for i in range(GD):
                for j in range(i, GD):
                    if i == j:
                        K[:, i::GD, i::GD] += D
                        K[:, i::GD, i::GD] += (mu + lam) * A[imap[(i, i)]]
                    else:
                        K[:, i::GD, j::GD] += lam * A[imap[(i, j)]]
                        K[:, i::GD, j::GD] += mu * A[imap[(i, j)]].transpose(0, 2, 1)

                        K[:, j::GD, i::GD] += lam * A[imap[(i, j)]].transpose(0, 2, 1)
                        K[:, j::GD, i::GD] += mu * A[imap[(i, j)]]
        if out is None:
            return K

    def assembly_cell_matrix_fast(self, space,
                                  trialspace=None, testspace=None, coefspace=None,
                                  index=np.s_[:], cellmeasure=None, out=None):
        """
        @brief 基于无数值积分的组装方式
        """
        lam = self.lam
        mu = self.mu
        c = self.c
        mesh = space[0].mesh
        ldof = space[0].number_of_local_dofs()
        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()

        meshtype = mesh.type

        if trialspace is None:
            trialspace = space[0]
            TAFtype = space[0].btype
            TAFdegree = space[0].p
            TAFldof = space[0].number_of_local_dofs()
        else:
            TAFtype = trialspace.btype
            TAFdegree = trialspace.p
            TAFldof = trialspace.number_of_local_dofs()

        if testspace is None:
            testspace = trialspace
            TSFtype = TAFtype
            TSFdegree = TAFdegree
            TSFldof = TAFldof
        else:
            TSFtype = testspace.btype
            TSFdegree = testspace.p
            TSFldof = testspace.number_of_local_dofs()

        if coefspace is None:
            coefspace = testspace
            COFtype = TSFtype
            COFdegree = TSFdegree
            COFldof = TSFldof
        else:
            COFtype = coefspace.btype
            COFdegree = coefspace.p
            COFldof = coefspace.number_of_local_dofs()

        Itype = self.type
        dataindex = Itype + "_" + meshtype + "_TAF_" + TAFtype + "_" + \
                    str(TAFdegree) + "_TSF_" + TSFtype + "_" + str(TSFdegree)

        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        if GD == 2:
            # 每个元组代表一个弹性张量的二阶导数的索引对
            idx = [(0, 0), (0, 1), (1, 1)]
            # 将 idx 中的元组映射到一个整数上
            imap = {(0, 0): 0, (0, 1): 1, (1, 1): 2}
        elif GD == 3:
            idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
            imap = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 1): 3, (1, 2): 4, (2, 2): 5}

        A = []

        NC = len(cellmeasure)

        if out is None:
            K = np.zeros((NC, GD * ldof, GD * ldof), dtype=np.float64)
        else:
            assert out.shape == (NC, GD * ldof, GD * ldof)
            K = out

        # 对于每一个设定的索引对，利用四边形积分公式和基函数的梯度来计算一个积分项
        glambda = mesh.grad_lambda()
        if c is None:
            A = [np.einsum('ijkl, c, ck, cl -> cij', data[dataindex], cellmeasure, glambda[..., i], glambda[..., j],
                           optimize=True) for i, j in idx]
        else:
            if callable(c):
                u = coefspace.interpolate(c)
                cell2dof = coefspace.cell_to_dof()
                c = u[cell2dof]
            if np.isscalar(c):
                A = [c * np.einsum('ijkl, c, ck, cl -> cij', data[dataindex], cellmeasure, glambda[..., i],
                                   glambda[..., j],
                                   optimize=True) for i, j in idx]
            elif c.shape == (NC,):
                A = [np.einsum('ijkl, c, ck, cl, c -> cij', data[dataindex], cellmeasure, glambda[..., i],
                               glambda[..., j], c,
                               optimize=True) for i, j in idx]
            elif c.shape == (NC, COFldof):
                dataindex += "_COF_" + COFtype + "_" + str(COFdegree)
                A = [np.einsum('ijmkl, c, ck, cl, cm -> cij', data[dataindex], cellmeasure, glambda[..., i],
                               glambda[..., j], c,
                               optimize=True) for i, j in idx]
            else:
                raise ValueError("coef 不支持该类型")

        D = 0
        for i in range(GD):
            D += mu * A[imap[(i, i)]]
        if space[0].doforder == 'sdofs':  # 标量自由度优先排序 
            for i in range(GD):
                for j in range(i, GD):
                    if i == j:
                        K[:, i * ldof:(i + 1) * ldof, i * ldof:(i + 1) * ldof] += D
                        K[:, i * ldof:(i + 1) * ldof, i * ldof:(i + 1) * ldof] += (mu + lam) * A[imap[(i, i)]]
                    else:
                        K[:, i * ldof:(i + 1) * ldof, j * ldof:(j + 1) * ldof] += lam * A[imap[(i, j)]]
                        K[:, i * ldof:(i + 1) * ldof, j * ldof:(j + 1) * ldof] += mu * A[imap[(i, j)]].transpose(0, 2,
                                                                                                                 1)
                        K[:, j * ldof:(j + 1) * ldof, i * ldof:(i + 1) * ldof] += lam * A[imap[(i, j)]].transpose(0, 2,
                                                                                                                  1)
                        K[:, j * ldof:(j + 1) * ldof, i * ldof:(i + 1) * ldof] += mu * A[imap[(i, j)]]
        elif space[0].doforder == 'vdims':
            for i in range(GD):
                for j in range(i, GD):
                    if i == j:
                        K[:, i::GD, i::GD] += D
                        K[:, i::GD, i::GD] += (mu + lam) * A[imap[(i, i)]]
                    else:
                        K[:, i::GD, j::GD] += lam * A[imap[(i, j)]]
                        K[:, i::GD, j::GD] += mu * A[imap[(i, j)]].transpose(0, 2, 1)

                        K[:, j::GD, i::GD] += lam * A[imap[(i, j)]].transpose(0, 2, 1)
                        K[:, j::GD, i::GD] += mu * A[imap[(i, j)]]

        if out is None:
            return K

    def assembly_cell_matrix_ref(self, space, index=np.s_[:], cellmeasure=None):
        pass