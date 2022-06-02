import numpy as np


def adaptive(mesh, interface):
    """
    @brief 生成自适应的界面拟合网格 
    """

    NN = mesh.number_of_nodes()
    node= mesh.entity('node')

    phi = interface(node)

    if np.all(phi < 0):
        raise ValueError('初始网格在界面围成区域的内部，需要更换一个可以覆盖界面的网格')

    # Step 1: 一致二分法加密网格
    while np.all(phi>0):
        mesh.uniform_bisect()
        node = mesh.entity('node')
        phi = np.append(phi, interface(node[NN:]))
        NN = mesh.number_of_nodes()

    # Step 2: 估计离散曲率

    return mesh


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from fealpy.mesh import TriangleMesh
    from fealpy.mesh import MeshFactory as MF
    from fealpy.geometry import CircleCurve

    box = [-1, 1]*2

    mesh = MF.boxmesh2d(box, nx=1, ny=1, meshtype='tri')
    interface = CircleCurve(radius=0.5)

    adaptive(mesh, interface)

    imesh = interface.init_mesh(100)

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    imesh.add_plot(axes, markersize=0, cellcolor='r')
    plt.show()
