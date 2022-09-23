import numpy as np
from fealpy.mesh import StructureQuadMesh
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=
        """
        在二维网格上用有限差分求解带 PML 层的 Maxwell 方程 
        """)

parser.add_argument('--NS',
        default=200, type=int,
        help='区域 x 和 y 方向的剖分段数， 默认为 200 段.')

parser.add_argument('--NP',
        default=50, type=int,
        help='PML 层的剖分段数， 默认为 50 段.')

parser.add_argument('--NT',
        default=4000, type=int,
        help='时间剖分段数， 默认为 4000 段.')

parser.add_argument('--ND',
        default=20, type=int,
        help='一个波长剖分的网格段数， 默认为 20 段.')

parser.add_argument('--R',
        default=0.5, type=int,
        help='网比， 默认为 0.5.')

parser.add_argument('--m',
        default=6, type=float,
        help='')

parser.add_argument('--sigma',
        default=100, type=float,
        help='最大电导率，默认取 100.')

args = parser.parse_args()


NS = args.NS
NP = args.NP
NT = args.NT
ND = args.ND
R = args.R
m = args.m
sigma = args.sigma

domain = [0, 1, 0, 1] # 原始区域
h = 1/NS
delta = h*NP

def sigma_x(p):
    x = p[..., 0]
    shape = p.shape[:-1]
    val = np.zeros(shape, dtype=np.float64)
    flag = x < 0 
    val[flag] = sigma*((0 - x[flag])/delta)**m
    flag = x > 1
    val[flag] = sigma*((x[flag] - 1)/delta)**m
    return val

def sigma_y(p):
    y = p[..., 1]
    shape = p.shape[:-1]
    val = np.zeros(shape, dtype=np.float64)
    flag = y < 0 
    val[flag] = sigma*((0 - y[flag])/delta)**m
    flag = y > 1
    val[flag] = sigma*((y[flag] - 1)/delta)**m
    return val


domain = [-delta, 1+delta, -delta, 1+delta] # 增加了 PML 层的区域
mesh = StructureQuadMesh(domain, nx=NS+NP, ny=NS+NP) # 建立结构网格对象

sx0 = mesh.interpolation(sigma_x, intertype='cell')
sy0 = mesh.interpolation(sigma_y, intertype='cell') 

sx1 = mesh.interpolation(sigma_x, intertype='edgey')
sy1 = mesh.interpolation(sigma_y, intertype='edgey')

sx2 = mesh.interpolation(sigma_x, intertype='edgex')
sy2 = mesh.interpolation(sigma_y, intertype='edgex')



Ez = np.zeros([2, NS, NS], dtype=np.float_)
Dz = np.zeros([2, NS, NS], dtype=np.float_)
Hx = np.zeros([2, NS, NS + 1], dtype=np.float_)
Hy = np.zeros([2, NS + 1, NS], dtype=np.float_)
Bx = np.zeros([2, NS, NS + 1], dtype=np.float_)
By = np.zeros([2, NS + 1, NS], dtype=np.float_)

for i in range(NT):

    Bx[1, :, 1:-1] = (2 - R * h * sigmaY_2[:, 1:-1]) / (2 + R * h * sigmaY_2[:, 1:-1]) \
                     * Bx[0, :, 1:-1] - R * 2 / (2 + R * h * sigmaY_2[:, 1:-1]) \
                     * (Ez[0, :, 1:] - Ez[0, :, 0:-1])

    By[1, 1:-1, :] = (2 - R * h * sigmaX_1[1:-1, :]) / (2 + R * h * sigmaX_1[1:-1, :]) \
                     * By[0, 1:-1, :] + R * 2 / (2 + R * h * sigmaX_1[1:-1, :]) \
                     * (Ez[0, 1:, :] - Ez[0, 0:-1, :])

    Hx[1, :, 1:-1] = Hx[0, :, 1:-1] + (2 + R * h * sigmaX_2[:, 1:-1]) / 2 * Bx[1, :, 1:-1] \
                     - (2 - R * h * sigmaX_2[:, 1:-1]) / 2 * Bx[0, :, 1:-1]

    Hy[1, 1:-1, :] = Hy[0, 1:-1, :] + (2 + R * h * sigmaY_1[1:-1, :]) / 2 * By[1, 1:-1, :] \
                     - (2 - R * h * sigmaY_1[1:-1, :]) / 2 * By[0, 1:-1, :]

    Dz[1, :, :] = (2 - R * h * sigmaX_0[:, :]) / (2 + R * h * sigmaX_0[:, :]) * Dz[0, :, :] \
                  + R * 2 / (2 + R * h * sigmaX_0[:, :]) \
                  * (Hy[1, 1:, :] - Hy[1, 0:-1, :] - Hx[1, :, 1:] + Hx[1, :, 0:-1])

    Ez[1, :, :] = (2 - R * h * sigmaY_0[:, :]) / (2 + R * h * sigmaY_0[:, :]) * Ez[0, :, :] \
                  + 2 / (2 + R * h * sigmaY_0[:, :]) * (Dz[1, :, :] - Dz[0, :, :])

    Ez[1, 100, 100] = np.sin(2 * np.pi * i * (R / ND))

    Bx[0] = Bx[1].copy()
    By[0] = By[1].copy()
    Hx[0] = Hx[1].copy()
    Hy[0] = Hy[1].copy()
    Dz[0] = Dz[1].copy()
    Ez[0] = Ez[1].copy()

    fig = plt.figure()
    plt.imshow(Ez[0], cmap='jet', vmin=0, vmax=1, extent=domain)
    plt.title('dt={}'.format(i))
    plt.colorbar()
    figname = "f" + ("%i" % (i)).zfill(4) + ".png"
    plt.savefig(fname=figname)
    plt.close(fig)
