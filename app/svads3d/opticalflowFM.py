from scipy.ndimage import sobel, gaussian_filter
from fealpy.experimental.backend import backend_manager as bm


def compute_image_gradients(I):
    """
    计算图像的x和y方向的梯度。
    """
    I_x = sobel(I, axis=1)  # x方向梯度
    I_y = sobel(I, axis=0)  # y方向梯度
    return I_x, I_y

def compute_G_and_b(I_x, I_y, delta_I, u_x, u_y, w_x, w_y):
    """
    计算G矩阵和b向量。
    G = Σ [ I_x^2  I_xI_y ]
              [ I_xI_y  I_y^2 ]
    
    b = Σ [ δI I_x ]
              [ δI I_y ]
    """
    G = bm.zeros((2, 2))
    b = bm.zeros((2, 1))

    for x in range(u_x - w_x, u_x + w_x + 1):
        for y in range(u_y - w_y, u_y + w_y + 1):
            I_x_val = I_x[y, x]
            I_y_val = I_y[y, x]
            delta_I_val = delta_I[y, x]
            
            # 计算 G 矩阵
            G[0, 0] += I_x_val * I_x_val
            G[0, 1] += I_x_val * I_y_val
            G[1, 0] += I_x_val * I_y_val
            G[1, 1] += I_y_val * I_y_val

            # 计算 b 向量
            b[0] += delta_I_val * I_x_val
            b[1] += delta_I_val * I_y_val
    
    return G, b

from scipy.ndimage import shift

def lucas_kanade_single_level(I, J, u_x, u_y, w_x, w_y, max_iter=1000, epsilon=1e-5):
    """
    Lucas-Kanade光流算法，支持浮点位移的匹配误差。
    """
    I_x, I_y = compute_image_gradients(I)
    d = bm.zeros((2, 1))  # 初始光流为 0

    for i in range(max_iter):
        # 使用亚像素级平移
        shifted_J = shift(J, shift=(-d[1, 0], -d[0, 0]), order=1)

        # 计算匹配误差
        delta_I = I - shifted_J

        # 计算G矩阵和b向量
        G, b = compute_G_and_b(I_x, I_y, delta_I, u_x, u_y, w_x, w_y)

        # 计算光流增量
        delta_d = bm.linalg.solve(G, b)

        # 更新光流向量
        d += delta_d

        # 判断是否收敛
        if bm.linalg.norm(delta_d) < epsilon:
            break
    
    return d[0, 0], d[1, 0]


def gaussian_pyramid(I, levels, sigma=1):
    """
    创建高斯金字塔。
    
    参数：
    - I: 输入图像
    - levels: 金字塔层数
    - sigma: 高斯平滑的标准差
    
    返回：
    - 金字塔图像列表
    """
    pyramid = [I]
    for _ in range(1, levels):
        I_smoothed = gaussian_filter(pyramid[-1], sigma=sigma)
        I_downsampled = I_smoothed[::2, ::2]  # 下采样
        pyramid.append(I_downsampled)
    return pyramid

def pyramidal_lucas_kanade(I, J, u_x, u_y, w_x, w_y, levels=3, sigma=1):
    """
    基于金字塔的Lucas-Kanade光流算法。
    
    参数：
    - I: 参考图像
    - J: 浮动图像
    - u_x, u_y: 待跟踪点的坐标
    - w_x, w_y: 邻域窗口大小
    - levels: 金字塔层数
    - sigma: 高斯平滑的标准差
    
    返回：
    - 最终光流 d_x, d_y
    """
    # 构建图像金字塔
    pyramid_I = gaussian_pyramid(I, levels, sigma=sigma)
    pyramid_J = gaussian_pyramid(J, levels, sigma=sigma)

    # 从顶层开始逐层计算光流
    d_x, d_y = 0, 0
    for level in range(levels - 1, -1, -1):
        I_level = pyramid_I[level]
        J_level = pyramid_J[level]
        
        # 将光流放大到当前层级
        d_x *= 2
        d_y *= 2
        
        # 计算光流时，根据当前层级缩小邻域窗口
        scaled_w_x = max(1, w_x // (2 ** level))
        scaled_w_y = max(1, w_y // (2 ** level))

        # 在当前层计算光流
        d_x, d_y = lucas_kanade_single_level(I_level, J_level, u_x // (2 ** level), u_y // (2 ** level), scaled_w_x, scaled_w_y)

        # 在当前层计算光流(没有按比例缩小邻域窗口)
        #d_x, d_y = lucas_kanade_single_level(I_level, J_level, u_x // (2 ** level), u_y // (2 ** level), w_x, w_y)

    return d_x, d_y

# 测试代码
if __name__ == "__main__":
    
    # 示例图像，替换成你要使用的实际图像
    #I = np.random.rand(64, 64)  # 参考图像（随机生成）
    bm.set_backend('numpy')
    I = bm.zeros((64, 64))  # 生成全零的矩阵
    I[30:34, 30:34] = 255  # 中央区域的4x4部分设为255，表示一个亮点

    J = bm.roll(I, 1, axis=1)  # 浮动图像，进行简单平移

    # 参数设置
    u_x, u_y = 32, 32  # 待跟踪点的坐标
    w_x, w_y = 5, 5    # 邻域窗口大小
    levels = 3         # 金字塔层数
    sigma = 1        # 高斯卷积的标准差

    # 计算光流
    d_x, d_y = pyramidal_lucas_kanade(I, J, u_x, u_y, w_x, w_y, levels=levels, sigma=sigma)

    print(f"计算出的光流: d_x = {d_x}, d_y = {d_y}")
