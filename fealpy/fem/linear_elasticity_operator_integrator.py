import numpy as np
from typing import Optional, Union, Tuple

class LinearElasticityOperatorIntegrator:
    def __init__(self, lam: float, mu: float, q: Optional[int]=None):
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

    def assembly_cell_matrix(self, space: Tuple, index=np.s_[:], 
                             cellmeasure: Optional[np.ndarray]=None, 
                             out: Optional[np.ndarray]=None) -> Optional[np.ndarray]:
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
        mesh = space[0].mesh
        ldof = space[0].number_of_local_dofs()
        p = space[0].p # 空间的多项式阶数
        GD = mesh.geo_dimension()
        q = self.q if self.q is not None else p+1


        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        if GD == 2:
            # 每个元组代表一个弹性张量的二阶导数的索引对
            idx = [(0, 0), (0, 1),  (1, 1)]
            # 将 idx 中的元组映射到一个整数上
            imap = {(0, 0):0, (0, 1):1, (1, 1):2}
        elif GD == 3:
            idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
            imap = {(0, 0):0, (0, 1):1, (0, 2):2, (1, 1):3, (1, 2):4, (2, 2):5}

        A = []

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

        D = 0
        for i in range(GD):
            D += mu*A[imap[(i, i)]]
        print("D_shape:", D.shape)
        print("D:\n", D)
        if space[0].doforder == 'sdofs': # 标量自由度优先排序 
            for i in range(GD):
                for j in range(i, GD):
                    if i == j:
                        K[:, i*ldof:(i+1)*ldof, i*ldof:(i+1)*ldof] += D 
                        K[:, i*ldof:(i+1)*ldof, i*ldof:(i+1)*ldof] += (mu + lam)*A[imap[(i, i)]]
                    else:
                        K[:, i*ldof:(i+1)*ldof, j*ldof:(j+1)*ldof] += lam*A[imap[(i, j)]] 
                        K[:, i*ldof:(i+1)*ldof, j*ldof:(j+1)*ldof] += mu*A[imap[(i, j)]].transpose(0, 2, 1)
                        K[:, j*ldof:(j+1)*ldof, i*ldof:(i+1)*ldof] += lam*A[imap[(i, j)]].transpose(0, 2, 1)
                        K[:, j*ldof:(j+1)*ldof, i*ldof:(i+1)*ldof] += mu*A[imap[(i, j)]]
        elif space[0].doforder == 'vdims':
            for i in range(GD):
                for j in range(i, GD):
                    if i == j:
                        K[:, i::GD, i::GD] += D 
                        K[:, i::GD, i::GD] += (mu + lam)*A[imap[(i, i)]]
                    else:
                        K[:, i::GD, j::GD] += lam*A[imap[(i, j)]] 
                        K[:, i::GD, j::GD] += mu*A[imap[(i, j)]].transpose(0, 2, 1)

                        K[:, j::GD, i::GD] += lam*A[imap[(i, j)]].transpose(0, 2, 1)
                        K[:, j::GD, i::GD] += mu*A[imap[(i, j)]]
        if out is None:
            return K


    def assembly_cell_matrix_fast(self, space, index=np.s_[:], cellmeasure=None):
        pass


    def assembly_cell_matrix_ref(self, space, index=np.s_[:], cellmeasure=None):
        pass
