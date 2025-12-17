from fealpy.backend import bm
from fealpy.mesh import TriangleMesh
from fealpy.mesher import NACA4Mesher


bm.set_backend("numpy")


if __name__ == '__main__':
    # === 参数设定 (NACA 4位数) ===
    m = 0.02  # 最大弯度
    p = 0.4  # 弯度位置
    t = 0.12  # 厚度
    c = 1.0  # 弦长
    alpha = 15.0  # 攻角
    N = 50

    # 翼型网格参数
    box = [-0.5, 3.5, -0.5, 0.8]
    h = 0.05
    # 无攻角奇异点
    # singular_points = bm.array([[0, 0], [1.00662, 0.0]], dtype=bm.float64)
    # 有攻角奇异点
    singular_points = bm.array([[0, 0], [0.97476, 0.260567]], dtype=bm.float64)
    hs = [h/10, h/20]  # 奇异点处网格尺寸

    mesher = NACA4Mesher(m, p, t, c, alpha, N, box, singular_points)
    mesh = mesher.init_mesh(h, hs, is_quad=0, thickness=h/5, ratio=2.4, size=h/50)
    mesh.to_vtk(fname=f'./data/naca4_m{m}_p{p}_t{t}_c{c}_alpha{alpha}.vtu')
