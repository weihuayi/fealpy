import numpy as np


class LagrangeCurve:
    """
    @brief Lagrange 插值曲线
    """

    def __init__(self, node):
        self.node = node
        self.p = len(node)-1
        self.ftype = node.dtype


    def geo_dimension(self):
        return self.node.shape[-1]

    def top_dimension(self):
        return 1

    def __call__(self, xi):
        """
        @brief 
        """

        if type(xi) in {int, float}:
            bc = np.array([1-xi, xi], dtype=self.ftype)
        else:
            n = len(xi)
            bc = np.zeros((n, 2), dtype=self.ftype)
            bc[:, 0] = 1 - xi
            bc[:, 1] = xi

        TD = bc.shape[-1] - 1 
        p = self.p
        multiIndex = self.multi_index_matrix(p)

        c = np.arange(1, p+1, dtype=np.int_)
        P = 1.0/np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(TD+1)
        phi = np.prod(A[..., multiIndex, idx], axis=-1) # (NQ, ldof)

        ps = np.einsum('qi, id->qd', phi, self.node)
        return ps


    def multi_index_matrix(self, p):
        """
        """
        ldof = p + 1
        multiIndex = np.zeros((ldof, 2), dtype=np.int_)
        multiIndex[:, 0] = np.arange(p, -1, -1)
        multiIndex[:, 1] = p - multiIndex[:, 0]
        return multiIndex

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    node = np.array([
        [20, 5], 
        [10, 20], 
        [40, 50], 
        [60, 5], 
        [70, 8], 
        [100,56],
        [50, 50], 
        [40,60]], dtype=np.float64)

    curve = LagrangeCurve(node)

    fig = plt.figure()
    axes = fig.gca()
    axes.plot(node[:, 0], node[:, 1], 'b-.')

    xi = np.linspace(0, 1, 1000)
    ps = curve(xi)
    axes.plot(ps[:, 0], ps[:, 1], 'r')
    plt.show()
