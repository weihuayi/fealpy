import argparse
from fealpy.backend import backend_manager as bm

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        光滑元方法求解多重调和方程
        """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help="默认后端为 numpy. 还可以选择 pytorch, jax, tensorflow 等")


args = parser.parse_args()
bm.set_backend(args.backend)

from fealpy.fem import MthLaplaceSmoothFEMModel
model = MthLaplaceSmoothFEMModel()
model.set_pde('biharm2d')
#model.set_pde('biharm3d')
model.set_init_mesh()
m = 1
TD = model.mesh.TD
model.set_smoothness(m=m)
model.set_order(p=2**TD*m+1)
model.solve.set('mumps')
#model.run['uniform_refine']()
model.run['one_step']()


