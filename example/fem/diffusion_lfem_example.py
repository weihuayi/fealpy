import argparse

from fealpy.backend import backend_manager as bm

parser = argparse.ArgumentParser(description="Solve diffusion equation using arbitrary order Lagrange finite element method")

parser.add_argument('--backend',
        default='numpy', type=str,
        help="Default backend is numpy. You can also choose pytorch, jax, tensorflow, etc.")


args = parser.parse_args()
bm.set_backend(args.backend)

from fealpy.fem import DiffusionLFEMModel

model = DiffusionLFEMModel()
model.set_pde(3)
model.set_init_mesh(meshtype='uniform_tri', nx=20, ny=20)
model.set_space_degree(2)
model.set_diffusion_coef_type(coeftype='continuous')
model.solve.set('mumps')
model.run['uniform_refine']()


