import numpy as np 
from numpy.linalg import norm, det, inv


class QualityMetric:
    def show(self, q):
        print("\n质量最大值:\t", max(q))
        print("质量最小值:\t", min(q))
        print("质量平均值:\t", np.mean(q))
        print("质量均方根:\t", np.sqrt(np.mean(q**2)))


class InverseMeanRatio(QualityMetric):
    def __init__(self, w):
        self.invw = inv(w)

    def quality(self, mesh):
        """
        @brief 计算每个网格单元顶点的质量
        """
        J = mesh.jacobian_matrix()
        T = J@self.invw 
        q = 0.5*norm(T, axis=(-2, -1))/det(T)
        return q


class TriRadiusRatio(QualityMetric):
    def quality(self, mesh, return_grad=False):
        """
        @brief 计算半径比质量

        mu = R/(2*r)
        p = l_0 + l_1 + l_2
        q = l_0 * l_1 * l_2
        """
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        NC = mesh.number_of_cells()

        localEdge = mesh.ds.local_edge()
        v = [node[cell[:, j], :] - node[cell[:, i], :] for i, j in localEdge]
        l2 = np.zeros((NC, 3))
        for i in range(3):
            l2[:, i] = np.sum(v[i]**2, axis=1)
        l = np.sqrt(l2)
        p = l.sum(axis=1)
        q = l.prod(axis=1)
        area = np.cross(v[1], v[2])/2
        quality = p*q/(16*area**2)

        if return_grad:
            grad = np.zeros((NC, 3, GD), dtype=mesh.ftype)
            grad[:, 0, :]  = (1/p/l[:, 1] + 1/l2[:, 1])[:, None]*(node[cell[:, 0]] - node[cell[:, 2]])
            grad[:, 0, :] += (1/p/l[:, 2] + 1/l2[:, 2])[:, None]*(node[cell[:, 0]] - node[cell[:, 1]])
            grad[:, 0, :] += 
        
        else:
            return quality

    def grad_quality(self):
        """
        @brief 计算
        """

        NC = self.number_of_cells()
        NN = self.number_of_nodes()
        node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')

        localEdge = self.ds.localEdge
        v = [node[cell[:, j], :] - node[cell[:, i], :] for i, j in localEdge]

        l2 = np.zeros((NC, 3))
        for i in range(3):
            l2[:, i] = np.sum(v[i]**2, axis=1)

        l = np.sqrt(l2)
        p = l.sum(axis=1, keepdims=True)
        q = l.prod(axis=1, keepdims=True)
        mu = p*q/(16*area**2)
        c = mu*(1/(p*l) + 1/l2)

        val = np.zeros((NC, 3, 3), dtype=sefl.ftype)
        val[:, 0, 0] = c[:, 1] + c[:, 2]
        val[:, 0, 1] = -c[:, 2]
        val[:, 0, 2] = -c[:, 1]

        val[:, 1, 0] = -c[:, 2]
        val[:, 1, 1] = c[:, 0] + c[:, 2]
        val[:, 1, 2] = -c[:, 0]

        val[:, 2, 0] = -c[:, 1]
        val[:, 2, 1] = -c[:, 0]
        val[:, 2, 2] = c[:, 0] + c[:, 1]

        I = np.broadcast_to(cell[:, None, :], shape=(NC, 3, 3))
        J = np.broadcast_to(cell[:, :, None], shape=(NC, 3, 3))
        A = csr_matrix((val, (I, J)), shape=(NN, NN))

        cn = mu/area
        val[:, 0, 0] = 0
        val[:, 0, 1] = -cn
        val[:, 0, 2] = cn

        val[:, 1, 0] = cn
        val[:, 1, 1] = 0
        val[:, 1, 2] = -cn

        val[:, 2, 0] = -cn
        val[:, 2, 1] = cn
        val[:, 2, 2] = 0
        B = csr_matrix((val, (I, J)), shape=(NN, NN))
        return A, B
