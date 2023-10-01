import numpy as np

import matplotlib.pyplot as plt

from scipy import ndimage

from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.signal import convolve2d


class TopologyOptimization:
    def __init__(self, nelx=60, nely=30):
        self.nelx = nelx
        self.nely = nely
        self.struc = np.ones((nely, nelx))
        self.lsf = self.reinit(self.struc)
        self.shapeSens = np.zeros((nely, nelx))
        self.topSens = np.zeros((nely, nelx))        
        self.KE, self.KTr, self.lambda_, self.mu = self.materialInfo(E, nu)



    def reinit(self, struc):
        strucFull = np.zeros((struc.shape[0] + 2, struc.shape[1] + 2))
        strucFull[1:-1, 1:-1] = struc

        dist_to_0 = ndimage.distance_transform_edt(strucFull)
        dist_to_1 = ndimage.distance_transform_edt(strucFull - 1)
        
        lsf = (~strucFull.astype(bool)).astype(int) * (dist_to_1 - 0.5) - strucFull * (dist_to_0 - 0.5)

        return lsf

    def stiffnessMatrix(self, k):
        K = np.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
        ])

        return K


    def materialInfo(self, E, nu):
        lambda_ = E * nu / ((1 + nu) * (1 - nu))
        mu = E / (2 * (1 + nu))
        k = [1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3 * nu/8,
            -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3 * nu/8]

        KE = E / (1 - nu**2) * stiffnessMatrix(k)

        k = [1/3, 1/4, -1/3, 1/4, -1/6, -1/4, 1/6, -1/4]
        KTr = E / (1 - nu) * stiffnessMatrix(k)

        return KE, KTr, lambda_, mu


nelx = 60
nely = 30
struc = np.ones((nely, nelx))


lsf = reinit(struc)

shapeSens = np.zeros((nely, nelx))
topSens = np.zeros((nely, nelx))

def stiffnessMatrix(k):
    # 根据给定的系数 k 形成刚度矩阵
    K = np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ])
    return K


def materialInfo(E, nu):
    # 设置材料参数并计算 Lamé 常数
    # E = 1.0 # 杨氏模量
    # nu = 0.3 # 泊松比
    lambda_ = E * nu / ((1 + nu) * (1 - nu))
    mu = E / (2 * (1 + nu))

    # 计算刚度矩阵 "KE"
    k = [1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3 * nu/8,
         -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3 * nu/8]
    KE = E / (1 - nu**2) * stiffnessMatrix(k)

    # 计算 "trace" 矩阵 "KTr"
    k = [1/3, 1/4, -1/3, 1/4, -1/6, -1/4, 1/6, -1/4]
    KTr = E / (1 - nu) * stiffnessMatrix(k)

    return KE, KTr, lambda_, mu


E = 1.0 # 杨氏模量
nu = 0.3 # 泊松比

# 设置刚度矩阵和其它材料参数
KE, KTr, lambda_, mu = materialInfo(E=E, nu=nu);


# FE 分析
def FE(struc, KE):
    nely, nelx = struc.shape
    # 初始化全局刚度矩阵，每个节点有 2 个自由度
    'K.shape = (2*(nelx+1)*(nely+1), 2*(nelx+1)*(nely+1)) = (56, 56)' 
    K = lil_matrix((2*(nelx+1)*(nely+1), 2*(nelx+1)*(nely+1)))
    
    # 初始化载荷向量
    'F.shape = (2*(nelx+1)*(nely+1), 1) = (56, 1)'
    F = lil_matrix((2*(nelx+1)*(nely+1), 1))
    
    # 初始化位移向量
    'U.shape = (2*(nelx+1)*(nely+1), ) = (56, )'
    U = np.zeros(2*(nelx+1)*(nely+1))

    # nelx*nely = 18 个单元
    for elx in range(nelx):
        for ely in range(nely):
            # 当前单元的左上角节点的全局索引
            n1 = (nely + 1) * elx + ely
            # 当前单元的右上角节点的全局索引
            n2 = (nely + 1) * (elx + 1) + ely
            '''
            {n1, n2} = (0, 4),   (1, 5),   (2, 6)
                       (4, 8),   (5, 8),   (6, 10)
                       (8, 12),  (9, 13),  (10, 14)
                       (12, 16), (13, 17), (14, 18)
                       (16, 20), (17, 21), (18, 22)
                       (20, 24), (21, 25), (22, 26)
            '''
            
            # edof 定义了当前单元的自由度
            # 对于一个包含 4 个节点的 2 维平面单元，每个节点有 2 个自由度，总共有 8 个自由度
            # 左下角开始的顺时针
            # [2*n1, 2*n1+1]:单元的左上角节点在全局位移向量中的 x 方向和 y 方向上的自由度的索引
            # [2*n2, 2*n2+1]:单元的右上角节点在全局位移向量中的 x 方向和 y 方向上的自由度的索引
            # [2*n2+2, 2*n2+3]:单元的右下角节点在全局位移向量中的 x 方向和 y 方向上的自由度的索引
            # [2*n1+2, 2*n1+3]:单元的左下角节点在全局位移向量中的 x 方向和 y 方向上的自由度的索引
            edof = [2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3]
            '''
            edof: [0, 1, 8, 9, 10, 11, 2, 3]
            edof: [2, 3, 10, 11, 12, 13, 4, 5]
            edof: [4, 5, 12, 13, 14, 15, 6, 7]
            edof: [8, 9, 16, 17, 18, 19, 10, 11]
            edof: [10, 11, 18, 19, 20, 21, 12, 13]
            edof: [12, 13, 20, 21, 22, 23, 14, 15]
            edof: [16, 17, 24, 25, 26, 27, 18, 19]
            edof: [18, 19, 26, 27, 28, 29, 20, 21]
            edof: [20, 21, 28, 29, 30, 31, 22, 23]
            edof: [24, 25, 32, 33, 34, 35, 26, 27]
            edof: [26, 27, 34, 35, 36, 37, 28, 29]
            edof: [28, 29, 36, 37, 38, 39, 30, 31]
            edof: [32, 33, 40, 41, 42, 43, 34, 35]
            edof: [34, 35, 42, 43, 44, 45, 36, 37]
            edof: [36, 37, 44, 45, 46, 47, 38, 39]
            edof: [40, 41, 48, 49, 50, 51, 42, 43]
            edof: [42, 43, 50, 51, 52, 53, 44, 45]
            edof: [44, 45, 52, 53, 54, 55, 46, 47]
            '''
            
            # 将单元刚度矩阵添加到全局刚度矩阵中
            K[np.ix_(edof, edof)] += max(struc[ely, elx], 0.0001) * KE

    #print("K:\n", K.toarray())
    
    # 定义载荷和支撑 - 桥梁
    # 载荷 F 在结构中心的一个节点上施加，并向下施加
    F[2 * (round(nelx/2) + 1) * (nely+1)] = 1
    # print("F:", F.shape)
    
    # fixeddofs 定义了固定的自由度
    fixeddofs = list( range( 2*(nely+1)-2, 2*(nely+1) ) ) + list( range( 2*(nelx+1)*(nely+1)-2, 2*(nelx+1)*(nely+1) ) )
    'fixeddofs: [6, 7, 54, 55]'
    
    # alldofs 定义了所有的自由度
    'len(alldofs) = 56'
    alldofs = list(range(2*(nely+1)*(nelx+1)))
    
    # freedofs 定义了非固定的自由度
    freedofs = list(set(alldofs) - set(fixeddofs))
    'len(freedofs) = 52'

    # 求解
    U[freedofs] = spsolve(csc_matrix(K[np.ix_(freedofs, freedofs)]), F[freedofs])

    return U


# 水平集函数的演化
def evolve(v, g, lsf, stepLength, w):
    # 为速度场 `v` 和源项 `g` 添加一个零边界
    # print("v:\n", v)
    # print("g:\n", g)
    vFull = np.pad(v, ((1,1),(1,1)), mode='constant', constant_values=0)
    gFull = np.pad(g, ((1,1),(1,1)), mode='constant', constant_values=0)
    # print("vFull:\n", vFull)
    # print("gFull:\n", gFull)
    
    # 基于 CFL 条件选择时间步长
    dt = 0.1 / np.max(np.abs(v))
    # print("时间步长 dt:", dt)

    # print("更新前的 lsf:", lsf)
    # 演化
    for i in range(int(10 * stepLength)):
        # 在离散的网格上估计导数，给出了 `lsf` 在各个方向上的变化率
        # `dpx` 水平方向上的正向差分，对应于 \frac{\partial\phi}{\partial x} 的向前差分近似
        dpx = np.roll(lsf, shift=-1, axis=1) - lsf
        # `dmx` 水平方向上的反向差分，对应于 \frac{\partial\phi}{\partial x} 的向后差分近似
        dmx = lsf - np.roll(lsf, shift=1, axis=1)
        # `dpy` 垂直方向上的正向差分，对应于 \frac{\partial\phi}{\partial y} 的向前差分近似
        dpy = np.roll(lsf, shift=-1, axis=0) - lsf
        # `dmy` 垂直方向上的反向差分，对应于 \frac{\partial\phi}{\partial y} 的向后差分近似
        dmy = lsf - np.roll(lsf, shift=1, axis=0)
        
        # 使用迎风差分格式更新水平集函数
        lsf = lsf - dt * np.minimum(vFull, 0) * np.sqrt( np.minimum(dmx, 0)**2 + np.maximum(dpx, 0)**2 + np.minimum(dmy, 0)**2 + np.maximum(dpy, 0)**2 ) \
                  - dt * np.maximum(vFull, 0) * np.sqrt( np.maximum(dmx, 0)**2 + np.minimum(dpx, 0)**2 + np.maximum(dmy, 0)**2 + np.minimum(dpy, 0)**2 ) \
                  - dt*w*gFull

    # print("更新后的 lsf:", lsf)
    # 从 `lsf` 获取新结构
    strucFULL = (lsf < 0).astype(int)
    # print("strucFULL:\n", strucFULL)
    struc = strucFULL[1:-1, 1:-1]
    # print("struc:\n", struc)
    
    return struc, lsf


def updateStep(lsf, shapeSens, topSens, stepLength, topWeight):
    # 平滑灵敏度
    # 中心元素权重为 2，四个临接元素权重为 1
    kernel = 1/6 * np.array([[0, 1, 0], 
                             [1, 2, 1], 
                             [0, 1, 0]])

    # 使用 numpy 的 pad 函数来替代 MATLAB 的 padarray
    # 'edge'模式对应于 MATLAB 中的 'replicate'
    # print("shapeSens:\n", shapeSens)
    padded_shape_sens = np.pad(shapeSens, ((1, 1), (1, 1)), mode='edge')
    # print("复制原 shapeSens 的第一行和最后一行、第一列和最后一列 padded_shape_sens", padded_shape_sens.shape, ":\n", padded_shape_sens)
    
    padded_top_sens = np.pad(topSens, ((1, 1), (1, 1)), mode='edge')
    # print("复制原 topSens 的第一行和最后一行、第一列和最后一列 padded_top_sens", padded_top_sens.shape, ":\n", padded_top_sens)

    # 使用 scipy 的 convolve2d 函数来进行卷积操作
    shape_sens_smoothed = convolve2d(padded_shape_sens, kernel, mode='valid')
    # print("平滑后的 shapeSens", shape_sens_smoothed.shape, ":\n", shape_sens_smoothed)
    
    top_sens_smoothed = convolve2d(padded_top_sens, kernel, mode='valid')
    # print("平滑后的 topSens", top_sens_smoothed.shape, ":\n", top_sens_smoothed)


    # Load bearing pixels must remain solid - Bridge
    # 对于一个桥梁结构，桥的两端和中间的承载部分是关键的，不能被移除，因此必须将这些部分的敏感度设置为 0，确保它们在后续的优化过程中不会被更改
    # shapeSens 和 topSens 的最后一行的第一个元素、中间两个元素以及最后一个元素设置为 0，这表示这些元素使承载载荷的，必须始终保持 solid，不允许在后续的设计更新中变为 void
    # print("第一个、中间两个和最后一个元素下标", [0, round((shapeSens.shape[1]-1)/2), round((shapeSens.shape[1]-1)/2) + 1, -1])
    
    shape_sens_smoothed[-1, [0, round((shapeSens.shape[1]-1)/2), round((shapeSens.shape[1]-1)/2) + 1, -1]] = 0
    # print("承载载荷保持 solid 后的 shape_sens_smoothed", shape_sens_smoothed.shape, ":\n", shape_sens_smoothed)

    top_sens_smoothed[-1, [0, round((topSens.shape[1]-1)/2), round((topSens.shape[1]-1)/2) + 1, -1]] = 0
    # print("承载载荷保持 solid 后的 top_sens_smoothed", top_sens_smoothed.shape, ":\n", top_sens_smoothed)

    # 通过 `evolve` 更新设计
    # 使用 `shapeSens` 的负数作为法向速度
    # 使用 `topSens * (lsf[1:-1, 1:-1] < 0)` 来计算 forcing term `g`
    struc, lsf = evolve(-shape_sens_smoothed, top_sens_smoothed * (lsf[1:-1, 1:-1] < 0), lsf, stepLength, topWeight)

    # 返回更新后的 struc 和 lsf
    return struc, lsf


Num = 10

# 初始化目标数组
objective = np.zeros(Num)

# 所需的体积分数
volReq = 0.3

# 设计更新中演化 LSF 的时间间隔
stepLength = 3

# 演化方程中 forcing 项的权重
topWeight = 2

# 水平集函数重新初始化为符号距离函数的频率
numReinit = 2

for iterNum in range(Num):
    # 有限元分析，计算灵敏度
    U = FE(struc, KE)
    for elx in range(nelx):
        for ely in range(nely):
            # 当前单元的左上角节点的全局索引
            n1 = (nely+1) * elx + ely
            # 当前单元的右上角节点的全局索引
            n2 = (nely+1) * (elx+1) + ely
            
            # 从全局位移向量 U 中提取当前单元的位移
            # 对于一个包含 4 个节点的 2 维平面单元，每个节点有 2 个自由度，总共有 8 个自由度
            Ue = U[np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3])]
            
            # 计算形状敏感度
            shapeSens[ely, elx] = -max(struc[ely, elx], 0.0001) * Ue.T @ KE @ Ue
            'shapeSens.shape = (nely, nelx)'
            
            # 计算拓扑敏感度
            coeff = np.pi/2 * (lambda_ + 2*mu) / mu / (lambda_ + mu)
            topSens[ely, elx] = struc[ely, elx] * coeff * (4*mu * Ue.T @ KE @ Ue) * (4*mu * Ue.T @ KE @ Ue + (lambda_ - mu) * Ue.T @ KTr @ Ue)
            'topSens.shape = (nely, nelx)'
            
    # print("Ue:\n", Ue)
    # print("目标函数的形状灵敏度 shapeSens:\n", shapeSens)
    # print("目标函数的拓扑灵敏度 topSens:\n", topSens)

    # 存储数据、打印和绘制信息
    # 存储目标值
    objective[iterNum] = -np.sum(shapeSens)

    # 计算当前的体积分数
    volCurr = np.sum(struc) / (nelx*nely)

    # 显示当前迭代的信息
    print(f'Iter: {iterNum+1}, Compl.: {objective[iterNum]:.4f}, Vol.: {volCurr:.3f}')
    
    # 绘制当前结构
    plt.imshow(-struc, cmap='gray', vmin=-1, vmax=0)
    plt.axis('off')
    plt.show()

    # 检查收敛性
    # 算法的前 5 次迭代不检查收敛性，检查当前体积 volCurr 与所需体积的差异，最近 5 次迭代的目标函数值与当前迭代的目标函数值的差异
    if iterNum > 5 and (abs(volCurr - volReq) < 0.005) and \
        np.all( abs(objective[-1] - objective[-6:-1]) < 0.01 * abs(objective[-1]) ):
        break

    # 设置增广 Lagrange 参数
    if iterNum == 1:
        # Lagrange 乘子
        la = -0.01
        # 惩罚因子
        La = 1000
        # 学习率
        alpha = 0.9
    else:
        la = -0.01
        La = 1000
        alpha = 0.9
        la = la - 1/La * (volCurr - volReq)
        La = alpha * La

    # 包含体积灵敏度
    shapeSens = shapeSens - la + 1/La * (volCurr-volReq)
    topSens = topSens + np.pi * ( la - 1/La * (volCurr-volReq) )
    # print("加入体积灵敏度后的目标函数的形状灵敏度 shapeSens:\n", shapeSens)
    # print("加入体积灵敏度后的目标函数的拓扑灵敏度 topSens:\n", topSens)

    # 设计更新
    struc, lsf = updateStep(lsf, shapeSens, topSens, stepLength, topWeight)

    # 对水平集函数进行重新初始化
    if iterNum % numReinit == 0:
        lsf = reinit(struc)

