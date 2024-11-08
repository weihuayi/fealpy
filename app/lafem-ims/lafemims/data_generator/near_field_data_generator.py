
import time
import numpy as np
from numpy.typing import NDArray
from math import sqrt

from fealpy.ml.generator import NearFieldDataFEMGenerator2d

start_time = time.time()

num_of_scatterers = 40000
n=64
domain = [-6, 6, -6, 6]
u_inc = 'cos(d_0*k*x + d_1*k*y) + sin(d_0*k*x + d_1*k*y) * 1j'
d = [[sqrt(0.5), sqrt(0.5)]]
k=[1, 3, 5]
M=20
data = np.load(f"data_m{M}_{num_of_scatterers}.npz")
c=data["c"]
origin_point = np.array([0.0, 0.0])
num_of_reciever_points = 29
reciever_points = np.load(f'./points_data/reciever_points_{num_of_reciever_points}.npy')

def lev_func(points: NDArray, complexity:int, coefficient:NDArray, origin:NDArray):

    points = points - origin
    angles_rad = np.zeros_like(points[:, 0], dtype=np.float64)  # 创建一个和点集大小相同的数组，用于存储角度弧度
    zero_indices = np.where(points[:, 0] == 0)  # 找到分母为零的索引
    non_zero_indices = np.where(points[:, 0] != 0)  # 找到分母不为零的索引
    
    # 处理分母为零的情况
    angles_rad[zero_indices] = np.pi / 2 if np.any(points[zero_indices, 1] > 0) else 3 * np.pi / 2
    
    # 处理分母不为零的情况
    slopes = points[non_zero_indices, 1] / points[non_zero_indices, 0]  # 计算斜率
    angle_rad = np.arctan(slopes)  # 计算角度弧度
    angles_rad[non_zero_indices] = np.real(angle_rad)
    
    # 将负值转换为正值
    negative_angle_indices = np.where(angles_rad < 0)
    angles_rad[negative_angle_indices] += np.pi
    
    # 调整角度弧度，确保在0到2*pi之间
    angles_rad = angles_rad % (2 * np.pi)
    
    # 处理负斜率的情况
    negative_slope_indices = np.where(points[:, 0] >= 0) and np.where(points[:, 1] < 0)
    angles_rad[negative_slope_indices] += np.pi
    
    r_t = coefficient[0]
    for i in range(complexity):
        r_t += coefficient[i+1] * np.cos((i+1) * angles_rad) + coefficient[i+M+1] * np.sin((i+1) * angles_rad)
    distances = r_t - np.linalg.norm(points, axis=1)
    flag = distances >= 0

    return flag

def main(scatterer_index, complexity, coefficient, origin):

    ind_func = lambda p: lev_func(p, complexity, coefficient, origin)

    generator = NearFieldDataFEMGenerator2d(domain=domain,
                                        mesh='UniformMesh',
                                        nx=100,
                                        ny=100,
                                        p=1,
                                        q=3,
                                        u_inc=u_inc,
                                        levelset=ind_func,
                                        d=d,
                                        k=k,
                                        reciever_points=reciever_points)
    generator.save(save_path=f'D:/ims_problem/m_{M}/k{k[0]}k{k[1]}k{k[2]}_{num_of_reciever_points}/', scatterer_index=scatterer_index)


if __name__ == "__main__":

    from multiprocessing import Pool

    pool = Pool(6)
    processes = []

    for idx in range(4500, num_of_scatterers):
        coefficient = c[idx, ...] # (NCir, GD+1)
        p = pool.apply_async(main, (idx, M, coefficient, origin_point))
        processes.append(p)

    pool.close()
    pool.join()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"生成数据时间: {elapsed_time} 秒")
