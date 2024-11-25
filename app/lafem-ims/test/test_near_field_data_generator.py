
from math import sqrt, pi
import numpy as np
from numpy.typing import NDArray

from lafemims.data_generator import NearFieldDataFEMGenerator2d
from fealpy.ml.sampler import CircleCollocator

# 定义计算域
domain = [-6, 6, -6, 6]

# 定义入射波函数
u_inc = 'cos(d_0*k*x + d_1*k*y) + sin(d_0*k*x + d_1*k*y) * 1j'

# 定义波矢量方向和波数
d = [[-sqrt(0.5), sqrt(0.5)]]
k = [2 * pi]

def levelset(p: NDArray, centers: NDArray, radius: NDArray) -> NDArray:
    """
    计算水平集函数值。
    
    参数:
    - p: 坐标点数组，形状为 (N, 2)
    - centers: 圆心坐标数组，形状为 (NCir, 2)
    - radius: 半径数组，形状为 (NCir,)
    
    返回:
    - 水平集函数值，形状为 (N,)
    """
    struct = p.shape[:-1]  # 获取输入点的结构形状
    p = p.reshape(-1, p.shape[-1])  # 将输入点展平为 (N, 2)
    
    # 计算每个点到所有圆心的距离
    dis = np.linalg.norm(p[:, None, :] - centers[None, :, :], axis=-1)  # 形状为 (N, NCir)
    
    # 计算每个点到最近圆的距离
    ret = np.min(dis - radius[None, :], axis=-1)  # 形状为 (N,)
    
    return ret.reshape(struct)  # 恢复输入点的原始结构形状

# 生成接收点
reciever_points = CircleCollocator(0, 0, 5).run(50)

# 定义圆的中心和半径
cirs = np.array([
    [0.4, -0.6, 0.2],
    [0.6, -0.5, 0.1],
    [0.3, 0.2, 0.3]
], dtype=np.float64)

centers = cirs[:, 0:2]  # 圆心坐标
radius = cirs[:, 2]     # 圆的半径

# 定义水平集函数
ls_fn = lambda p: levelset(p, centers, radius)

# 创建近场数据生成器
generator = NearFieldDataFEMGenerator2d(
    domain=domain,
    mesh='UniformMesh',
    nx=100,
    ny=100,
    p=1,
    q=3,
    u_inc=u_inc,
    levelset=ls_fn,
    d=d,
    k=k,
    reciever_points=reciever_points
)

# 可视化近场数据
generator.visualization_of_nearfield_data(k=k[0], d=d[0])
