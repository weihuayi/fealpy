
from fealpy.backend import backend_manager as bm
from fealpy.ml.sampler import CircleCollocator

from lafemims.data_generator import NearFieldDataFEMGenerator2d
from functional import levelset_fourier, genarate_scatterers_fourier


# 设置随机种子
SEED = 2025

# 定义计算域
domain = [-6, 6, -6, 6]

# 定义入射波函数
u_inc = 'cos(d_0*k*x + d_1*k*y) + sin(d_0*k*x + d_1*k*y) * 1j'

# 定义波矢量方向和波数
d = [[-bm.sqrt(0.5), bm.sqrt(0.5)]]
k = [2 * bm.pi]

#选择散射体复杂度、散射体中心点以及生成散射体个数
M = 20
origin_point = bm.array([0.0, 0.0])
num_of_scatterers = 40000

#确定外层接收点
num_of_reciever_points = 50
reciever_points = CircleCollocator(0, 0, 5).run(num_of_reciever_points)

#选择某个散射体
idx = 0
coefficients = genarate_scatterers_fourier(M, num_of_scatterers, SEED)     # shape == [2 * M + 1]

# 定义指示函数
ind_func = lambda p: levelset_fourier(p, M, coefficients[idx, ...], origin_point)

# 创建近场数据生成器
generator = NearFieldDataFEMGenerator2d(
    domain=domain,
    mesh='UniformMesh',
    nx=100,
    ny=100,
    p=1,
    q=3,
    u_inc=u_inc,
    levelset=ind_func,
    d=d,
    k=k,
    reciever_points=reciever_points
)

# 可视化近场数据
generator.visualization_of_nearfield_data(k=k[0], d=d[0])
