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

parser.add_argument('--area',
                    default=10.3, type=float,
                    help="Cross-sectional area of the beam, default is 10.3 m^2.")

parser.add_argument('--load',
                    default=-25000, type=float,
                    help="Distributed load, default is -25000 N/m.")

parser.add_argument('--length',
                    default=10, type=float,
                    help="Length of the beam, default is 10 m.")

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
