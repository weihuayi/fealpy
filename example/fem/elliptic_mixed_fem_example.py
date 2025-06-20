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

from fealpy.fem import EllipticMixedFEMModel

model = EllipticMixedFEMModel()
model.set_pde("poly")  
model.set_init_mesh(nx=10, ny=10)
model.set_space_degree(p=0)
model.space.set('rt')
model.solve.set('direct')
model.run['uniform_refine']()