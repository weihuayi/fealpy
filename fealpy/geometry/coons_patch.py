import numpy as np


def coons_patch(edge_ab, edge_bc, edge_cd, edge_da, n, m):
    # 获取角点
    a = edge_ab[0]
    b = edge_bc[0]
    c = edge_cd[0]
    d = edge_da[0]

    # 创建网格
    u = np.linspace(0, 1, m).reshape(-1, 1)
    v = np.linspace(0, 1, n).reshape(1, -1)

    # 边界点插值
    edge_ab_points = np.array(edge_ab)
    edge_cd_points = np.array(edge_cd)[::-1]
    edge_da_points = np.array(edge_da)[::-1]
    edge_bc_points = np.array(edge_bc)

    # 计算双线性混合
    bilinear_blend = (
        np.einsum('mn,nd->mnd', (1 - u), edge_ab_points)
        + np.einsum('mn,nd->mnd', u, edge_cd_points)
        + np.einsum('mn,md->mnd', (1 - v), edge_da_points)
        + np.einsum('mn,md->mnd', v, edge_bc_points)
        - np.einsum('mn,mn,d->mnd', (1 - u), (1 - v), a)
        - np.einsum('mn,mn,d->mnd', u, (1 - v), d)
        - np.einsum('mn,mn,d->mnd', (1 - u), v, b)
        - np.einsum('mn,mn,d->mnd', u, v, c)
    )

    return bilinear_blend