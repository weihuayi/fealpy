
from fealpy.backend import backend_manager as bm
from fealpy.ml.sampler import CircleCollocator

from lafemims.data_generator import NearFieldDataFEMGenerator2d
from functional import levelset_circles, genarate_scatterers_circles


# 设置随机种子
SEED = 2024

# 定义计算域
domain = [-6, 6, -6, 6]

# 定义入射波函数
u_inc = 'cos(d_0*k*x + d_1*k*y) + sin(d_0*k*x + d_1*k*y) * 1j'

# 定义波矢量方向和波数
d = [[-bm.sqrt(0.5), bm.sqrt(0.5)]]
k = [2 * bm.pi]

#散射体个数
num_of_scatterers = 40000
# 生成接收点
num_of_reciever_points = 50
reciever_points = CircleCollocator(0, 0, 5).run(num_of_reciever_points)

#选择某个散射体
idx = 0
centers, radius = genarate_scatterers_circles(num_of_scatterers, SEED)

# 定义水平集函数
ls_fn = lambda p: levelset_circles(p, centers[idx:idx+1, ...], radius[idx:idx+1, ...])

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
