import numpy as np

def sensitivity_filter(self, rmin, rho, dc):
    """
    应用 mesh-independency filter 进行每个单元的灵敏度过滤.

    Parameters:
    - rmin (float): Filter 半径.
    - rho (ndarray - (nely, nelx) ): 密度分布矩阵.
    - dc (ndarray - (nely, nelx) ): 原始的灵敏度矩阵.

    Returns:
    - dcn(ndarray - (nely, nelx) ): 过滤后的灵敏度矩阵.
    """

    nely, nelx = rho.shape
    dcn = np.zeros((nely, nelx))

    # 计算过滤器半径
    r = int(rmin)

    for i in range(nelx):
        for j in range(nely):
            sum_val = 0.0
            # 确定邻域的范围
            min_x = max(i - r, 0)
            max_x = min(i + r + 1, nelx)
            min_y = max(j - r, 0)
            max_y = min(j + r + 1, nely)

            for k in range(min_x, max_x):
                for l in range(min_y, max_y):

                    # Calculate convolution operator value for the element (k,l) with respect to (i,j)
                    fac = rmin - np.sqrt((i - k)**2 + (j - l)**2)

                    # Accumulate the convolution sum
                    sum_val += max(0, fac)

                    # 基于 convolution 算子的值修改单元的灵敏度
                    dcn[j, i] += max(0, fac) * rho[l, k] * dc[l, k]

            # Normalize the modified sensitivity for element (i,j)
            dcn[j, i] /= (rho[j, i] * sum_val)

    return dcn