import argparse

# 参数解析
parser = argparse.ArgumentParser(description="""
        Compute beam displacement using the linear finite element method.
        """)

parser.add_argument('--modulus',
                    default=200e9, type=float,
                    help="Young's modulus of the beam, default is 200e9 Pa.")

parser.add_argument('--inertia',
                    default=118.6e-6, type=float,
                    help="Moment of inertia of the beam, default is 118.6e-6 m^4.")

parser.add_argument('--pbar_log',
                    default=False, action='store_true',
                    help="Show progress bar log.")

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help="Log level, default is INFO.")

parser.add_argument('--beam_type',
                    default='euler_bernoulli_2d', type=str,
                    help='Type of beam, options: "euler_bernoulli_2d", "normal_2d", "euler_bernoulli_3d".')

parser.add_argument('--pde',
                    default=1, type=int,
                    help="Index of the beam model, default is 1.")

# 解析参数
options = vars(parser.parse_args())

from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

from fealpy.csm.fem import EulerBernoulliBeamFEMModel
beammodel = EulerBernoulliBeamFEMModel(options)
uh = beammodel.run()
# 打印结果
print("位移解向量 (uh):")
print(uh)
# 显示位移分布
beammodel.show(x=uh)
h = beammodel.pde.h
strain1,strain2=beammodel.material.compute_strain(u=uh)
print("节点上表面应变:", strain1)
print("节点下表面应变:", strain2)
# 显示应变分布
beammodel.show.set('strain')
beammodel.show(strain_top=strain1, strain_bottom=strain2)
stress1,stress2=beammodel.material.compute_stress(strain_top=strain1, strain_bottom=strain2)
beammodel.show.set('stress')
print("节点上表面应力:", stress1)
print("节点下表面应力:", stress2)    
# 显示应力分布   
beammodel.show(stress_top=stress1, stress_bottom=stress2)

