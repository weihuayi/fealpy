
from numpy.typing import NDArray
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

#多圆形并集-型 散射体
def levelset_circles(p: NDArray, centers: NDArray, radius: NDArray) -> NDArray:
    """
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
    dis = bm.linalg.norm(p[:, None, :] - centers[None, :, :], axis=-1)  # 形状为 (N, NCir)
    
    # 计算每个点到最近圆的距离
    ret = bm.min(dis - radius[None, :], axis=-1)  # 形状为 (N,)
    
    return ret.reshape(struct)  # 恢复输入点的原始结构形状

#半径函数为傅里叶系数-型 散射体
def levelset_fourier(points: NDArray, complexity: int, coefficient: NDArray, origin: NDArray) -> NDArray:
    """
    参数:
    - points: NDArray, 形状为 (N, 2)，表示 N 个点的坐标。
    - complexity: int, 水平集函数的复杂度。
    - coefficient: NDArray, 形状为 (2 * M + 1,)，表示傅里叶系数。
    - origin: NDArray, 形状为 (2,)，表示原点坐标。

    返回:
    - flag: NDArray, 形状为 (N,)，表示每个点是否在水平集函数内。
    """
    points = points - origin
    angles_rad = bm.zeros_like(points[:, 0], dtype=bm.float64)  # 创建一个和点集大小相同的数组，用于存储角度弧度

    # 处理分母为零的情况
    zero_indices = bm.where(points[:, 0] == 0)
    angles_rad[zero_indices] = bm.pi / 2 if bm.any(points[zero_indices, 1] > 0) else 3 * bm.pi / 2

    # 处理分母不为零的情况
    non_zero_indices = bm.where(points[:, 0] != 0)
    slopes = points[non_zero_indices, 1] / points[non_zero_indices, 0]  # 计算斜率
    angles_rad[non_zero_indices] = bm.arctan(slopes)  # 计算角度弧度

    # 将负值转换为正值
    negative_angle_indices = bm.where(angles_rad < 0)
    angles_rad[negative_angle_indices] += bm.pi

    # 调整角度弧度，确保在0到2*pi之间
    angles_rad = angles_rad % (2 * bm.pi)

    # 处理负斜率的情况
    negative_slope_indices = bm.where((points[:, 0] >= 0) & (points[:, 1] < 0))
    angles_rad[negative_slope_indices] += bm.pi

    r_t = coefficient[0]
    
    for i in range(complexity):
        r_t += coefficient[i + 1] * bm.cos((i + 1) * angles_rad) + coefficient[i + complexity + 1] * bm.sin((i + 1) * angles_rad)

    distances = r_t - bm.linalg.norm(points, axis=1)
    flag = distances >= 0

    return flag

#################################################################################################################################

def generate_scatterers_circles(num_of_scatterers: int, seed: int) ->TensorLike:
    """
    参数:
    - num_of_scatterers: int, 散射体的数量。
    
    返回:
    - ctrs: TensorLike, 形状为 (num_of_scatterers, 2)，表示每个散射体的中心坐标。
    - rads: TensorLike, 形状为 (num_of_scatterers,)，表示每个散射体的半径。
    """
    bm.random.seed(seed)
    
    ctrs = bm.random.rand(num_of_scatterers, 2) * 1.6 - 0.8 # (NCir, GD)
    b = bm.min(0.9-bm.abs(ctrs), axis=-1) # (NCir, )
    rads = bm.random.rand(num_of_scatterers) * (b-0.1) + 0.1 # (NCir, )
    ctrs = bm.astype(ctrs, bm.float64)
    rads = bm.astype(rads, bm.float64)

    return ctrs, rads
def generate_scatterers_fourier(complexity: int, num_of_scatterers: int, seed: int) ->TensorLike:
    """
    参数:
    - complexity: int, 水平集函数的复杂度。
    - num_of_scatterers: int, 散射体的数量。
    
    返回:
    - c: TensorLike, 形状为 (num_of_scatterers, 2*complexity+1)，表示每个散射体的参数。
    """
    bm.random.seed(seed)

    c = bm.zeros((num_of_scatterers, 2*complexity+1), dtype=bm.float64)
    radius = bm.random.uniform(0, 0.1, (num_of_scatterers, complexity))
    theta = bm.random.uniform(0, 2*bm.pi, (num_of_scatterers, complexity))

    c[:, 0:1] = bm.random.uniform(1, 1.2, (num_of_scatterers, 1))
    c[:, 1:complexity+1] = radius * bm.cos(theta)
    c[:, complexity+1: ] = radius * bm.sin(theta)

    return c
