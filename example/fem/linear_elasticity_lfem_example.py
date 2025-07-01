import argparse
from fealpy.backend import backend_manager as bm

## 参数解析
parser = argparse.ArgumentParser(description=
    """Solve 3D linear elasticity problem using linear Lagrange finite element method""")
parser.add_argument('--backend',
        default='numpy', type=str,
        help="Default backend is numpy. Other options: pytorch, jax, tensorflow, etc.")
args = parser.parse_args()

bm.set_backend(args.backend)

from fealpy.fem.linear_elasticity_lfem_model import  LinearElasticityLFEMModel
model = LinearElasticityLFEMModel()

model.set_pde("boxpoly")

model.set_init_mesh(meshtype='hex')

model.set_space_degree(p=1)

model.run['uniform_refine']()
print("-----------------------------")