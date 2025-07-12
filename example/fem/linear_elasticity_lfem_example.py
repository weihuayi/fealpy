import argparse
from fealpy.backend import backend_manager as bm
from fealpy.model.linear_elasticity.box_poly_data_3d import BoxPolyData3d

<<<<<<< Updated upstream
=======
## 参数解析
>>>>>>> Stashed changes
parser = argparse.ArgumentParser(description=
    """Solve 3D linear elasticity problem using linear Lagrange finite element method""")
parser.add_argument('--backend',
        default='numpy', type=str,
        help="Default backend is numpy. Other options: pytorch, jax, tensorflow, etc.")
args = parser.parse_args()

bm.set_backend(args.backend)

from fealpy.fem.linear_elasticity_lfem_model import  LinearElasticityLFEMModel
model = LinearElasticityLFEMModel()

<<<<<<< Updated upstream
# pde = BoxPolyData3d(box=[0, 2, 0, 2, 0, 2])
# model.set_pde(pde)  
model.set_pde("boxpoly3d")
# model.set_pde("boxpoly2d")  
# model.set_pde("boxsinsin2d")  

#model.set_init_mesh(meshtype='uniform_tri')
model.set_init_mesh(meshtype='custom_hex')
# model.set_init_mesh(meshtype='uniform_tet', nx=10, ny=10, nz=10)
=======
model.set_pde("boxpoly")

model.set_init_mesh(meshtype='hex')
>>>>>>> Stashed changes

model.set_space_degree(p=1)

model.run['uniform_refine']()
print("-----------------------------")