
import numpy as np
from typing import Optional, Union, Tuple

class ProvidesSymmetricTangentOperatorIntegrator:
    def __init__(self, D, q: Optional[int]=None):
        """
        初始化 ProvidesSymmetricTangentOperatorIntegrator 类

        参数:
        D : 切算子矩阵
        q (Optional[int]): 积分阶次，默认为 None
        """
        self._D = D
        self.q = q

    def assembly_cell_matrix(self, space: Tuple, index=np.s_[:], 
                             cellmeasure: Optional[np.ndarray]=None, 
                             out: Optional[np.ndarray]=None) -> Optional[np.ndarray]:
        """
        构建切算子有限元矩阵

        参数:
        space (Tuple): 有限元空间
        index (Union[np.s_, np.ndarray]): 选定的单元索引，默认为全部单元
        cellmeasure (Optional[np.ndarray]): 对应单元的度量，默认为 None
        out (Optional[np.ndarray]): 输出矩阵，默认为 None

        返回:
        Optional[np.ndarray]: 如果 out 参数为 None，则返回线性弹性有限元矩阵，否则不返回
        """
        self.space = space[0]
        self.mesh = space[0].mesh
        mesh = self.mesh
        ldof = space[0].number_of_local_dofs()
        p = space[0].p # 空间的多项式阶数
        GD = mesh.geo_dimension()
        q = self.q if self.q is not None else p+1

        if GD == 2:
            # 每个元组代表一个弹性张量的二阶导数的索引对
            idx = [(0, 0), (0, 1),  (1, 1)]
            # 将 idx 中的元组映射到一个整数上
            imap = {(0, 0):0, (0, 1):1, (1, 1):2}
        elif GD == 3:
            idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
            imap = {(0, 0):0, (0, 1):1, (0, 2):2, (1, 1):3, (1, 2):4, (2, 2):5}


        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        A = []
        D = self._D

        qf =  mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        grad = space[0].grad_basis(bcs, index=index) # (NQ, NC, ldof, GD)

        NC = len(cellmeasure)

        if out is None:
            K = np.zeros((NC, GD*ldof, GD*ldof), dtype=np.float64)
        else:
            assert out.shape == (NC, GD*ldof, GD*ldof)
            K = out

        # 对于每一个设定的索引对，利用四边形积分公式和基函数的梯度来计算一个积分项
        A = [np.einsum('i, ijm, ijn, j->jmn', ws, grad[..., i], grad[..., j], 
                       cellmeasure, optimize=True) for i, j in idx]
        d00 = D[..., 0, 0]
        d01 = D[..., 0, 1]
        d02 = D[..., 0, 2]
        d10 = D[..., 1, 0]
        d11 = D[..., 1, 1]
        d12 = D[..., 1, 2]
        d20 = D[..., 2, 0]
        d21 = D[..., 2, 1]
        d22 = D[..., 2, 2]

        # 默认标量自由度排序优先
        if space[0].doforder == 'sdofs': # 标量自由度优先排序 
            K[:, 0:ldof, 0:ldof] += np.einsum('i,ijm->ijm', d00, A[imap[(0, 0)]])
            K[:, 0:ldof, 0:ldof] += np.einsum('i,ijm->ijm', d20, A[imap[(0, 1)]])
            K[:, 0:ldof, 0:ldof] += np.einsum('i,ijm->ijm', d02, A[imap[(0, 1)]].transpose(0, 2, 1))
            K[:, 0:ldof, 0:ldof] += np.einsum('i,ijm->ijm', d22, A[imap[(1, 1)]])
            
            K[:, 0:ldof, ldof:2*ldof] += np.einsum('i,ijm->ijm', d01, A[imap[(0, 1)]])
            K[:, 0:ldof, ldof:2*ldof] += np.einsum('i,ijm->ijm', d21, A[imap[(1, 1)]])
            K[:, 0:ldof, ldof:2*ldof] += np.einsum('i,ijm->ijm', d22, A[imap[(0, 1)]].transpose(0, 2, 1))
            K[:, 0:ldof, ldof:2*ldof] += np.einsum('i,ijm->ijm', d02, A[imap[(0, 0)]])
            
            K[:, ldof:2*ldof, 0:ldof] += np.einsum('i,ijm->ijm', d10, A[imap[(0, 1)]].transpose(0, 2, 1))
            K[:, ldof:2*ldof, 0:ldof] += np.einsum('i,ijm->ijm', d20, A[imap[(0, 0)]])
            K[:, ldof:2*ldof, 0:ldof] += np.einsum('i,ijm->ijm', d12, A[imap[(1, 1)]])
            K[:, ldof:2*ldof, 0:ldof] += np.einsum('i,ijm->ijm', d22, A[imap[(0, 1)]])
            
            K[:, ldof:2*ldof, ldof:2*ldof] += np.einsum('i,ijm->ijm', d11, A[imap[(1, 1)]])
            K[:, ldof:2*ldof, ldof:2*ldof] += np.einsum('i,ijm->ijm', d21, A[imap[(0, 1)]])
            K[:, ldof:2*ldof, ldof:2*ldof] += np.einsum('i,ijm->ijm', d12, A[imap[(0, 1)]].transpose(0, 2, 1))
            K[:, ldof:2*ldof, ldof:2*ldof] += np.einsum('i,ijm->ijm', d22, A[imap[(0, 0)]])
        elif space[0].doforder == 'vdims':
            K[:, 0::GD, 0::GD] += np.einsum('i,ijm->ijm', d00, A[imap[(0, 0)]])
            K[:, 0::GD, 0::GD] += np.einsum('i,ijm->ijm', d02, A[imap[(0, 1)]])
            K[:, 0::GD, 0::GD] += np.einsum('i,ijm->ijm', d20, A[imap[(0, 1)]].transpose(0, 2, 1))
            K[:, 0::GD, 0::GD] += np.einsum('i,ijm->ijm', d22, A[imap[(1, 1)]])
            
            K[:, 0::GD, 1::GD] += np.einsum('i,ijm->ijm', d01, A[imap[(0, 1)]])
            K[:, 0::GD, 1::GD] += np.einsum('i,ijm->ijm', d21, A[imap[(1, 1)]])
            K[:, 0::GD, 1::GD] += np.einsum('i,ijm->ijm', d22, A[imap[(0, 1)]].transpose(0, 2, 1))
            K[:, 0::GD, 1::GD] += np.einsum('i,ijm->ijm', d02, A[imap[(0, 0)]])
            
            K[:, 1::GD, 0::GD] += np.einsum('i,ijm->ijm', d10, A[imap[(0, 1)]].transpose(0, 2, 1))
            K[:, 1::GD, 0::GD] += np.einsum('i,ijm->ijm', d12, A[imap[(1, 1)]])
            K[:, 1::GD, 0::GD] += np.einsum('i,ijm->ijm', d22, A[imap[(0, 1)]])
            K[:, 1::GD, 0::GD] += np.einsum('i,ijm->ijm', d20, A[imap[(0, 0)]])
            
            K[:, 1::GD, 1::GD] += np.einsum('i,ijm->ijm', d11, A[imap[(1, 1)]])
            K[:, 1::GD, 1::GD] += np.einsum('i,ijm->ijm', d21, A[imap[(0, 1)]])
            K[:, 1::GD, 1::GD] += np.einsum('i,ijm->ijm', d12, A[imap[(0, 1)]].transpose(0, 2, 1))
            K[:, 1::GD, 1::GD] += np.einsum('i,ijm->ijm', d22, A[imap[(0, 0)]])


        if out is None:
            return K


    def assembly_cell_matrix_fast(self, space0, _, index=np.s_[:], cellmeasure=None):
        pass


    def assembly_cell_matrix_ref(self, space0, _, index=np.s_[:], cellmeasure=None):
        pass
