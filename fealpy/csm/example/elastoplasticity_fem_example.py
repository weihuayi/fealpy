import argparse
from fealpy.backend import backend_manager as bm
from fealpy.csm.fem import ElastoplasticityFEMModel

# 参数解析
parser = argparse.ArgumentParser(description="""
        用有限元方法计算弹塑性问题的位移
        """)

parser.add_argument('--pde',
                    default='1', type=str,
                    help='选择预设的弹塑性问题示例，默认为"1"')

parser.add_argument('--space_degree',
                    default=1, type=int,
                    help='选择有限元空间的多项式阶数，默认为1')

parser.add_argument('--pbar_log',
                    default=False, action='store_true',
                    help='是否显示进度条日志.')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                        help='日志级别, 默认为 INFO.')
# 解析参数
options = vars(parser.parse_args())

bm.set_backend('numpy')

model = ElastoplasticityFEMModel(options)
mesh = model.mesh
# 测试有限元空间
model.set_space()
# 测试材料参数设置
model.set_space_degree(options['space_degree'])
# 测试材料模型
model.set_material_parameters(E=2.069e5, nu=0.29)  # 设置材料参数
# 测试应力计算
uh = model.space.function()
cell2dof = model.space.cell_to_dof()
uh_cell = uh[cell2dof]
qf = mesh.quadrature_formula(q=model.space.scalar_space.p+3)
bcs, ws = qf.get_quadrature_points_and_weights()
NQ = bcs.shape[0]
NC = mesh.number_of_cells()
plastic_strain = bm.zeros((NC, NQ, 3), dtype=bm.float64)
stress = model.compute_stress(displacement=bm.zeros_like(uh), plastic_strain=plastic_strain)
# 测试内部力积分子
'''
from fealpy.csm.fem import ElastoplasticitySourceIntIntegrator
from fealpy.fem import LinearForm
lfrom = LinearForm(model.space)
lfrom.add_integrator(
    ElastoplasticitySourceIntIntegrator(
        strain_matrix=model.B,  # 使用模型的应变矩阵
        stress=stress,  # 使用计算得到的应力
        q=model.space.scalar_space.p+3  # 使用适当的积分点数
    )
)
F = lfrom.assembly()  # 组装内部力项
print("内部力项 F:", F)
print(F.shape)
'''
# 测试刚度矩阵
from fealpy.csm.fem import ElastoplasticDiffusionIntegrator
from fealpy.fem import BilinearForm
BilinearForm = BilinearForm(model.space)
BilinearForm.add_integrator(
    ElastoplasticDiffusionIntegrator(
        D_ep=model.D,  # 使用弹塑性矩阵
        material=model.pfcm,  # 使用模型的材料参数
        q=model.space.scalar_space.p+3  # 使用适当的积分点数
    )
)
K = BilinearForm.assembly()  # 组装刚度矩阵
print("刚度矩阵 K:", K)


'''
# 测试网格可视化
from matplotlib import pyplot as plt
fig = plt.figure()
axes = fig.add_subplot(111)
mesh.add_plot(axes) # 画出网格背景
mesh.find_cell(axes, showindex=True) # 找到单元重心
plt.show()
'''