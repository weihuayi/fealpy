import argparse

from fealpy.backend import backend_manager as bm

## 参数解析
parser = argparse.ArgumentParser(description=
    """
    最低阶Raviart-Thomas元和分片常数空间混合元求解椭圆方程
    """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help="默认后端为 numpy. 还可以选择 pytorch, jax, tensorflow 等")


args = parser.parse_args()
bm.set_backend(args.backend)

from fealpy.csm.fem import EllipticMixedFEMModel

model = EllipticMixedFEMModel()
model.set_pde()
model.set_init_mesh(nx=20, ny=20)
model.set_order(p=0)
model.solve.set('direct')
model.run['uniform_refine']()