import argparse


# Argument parsing
parser = argparse.ArgumentParser(description=
        """
        Finite Element Solution for 3D Timoshenko Beam.
        """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help='Default backend is numpy')

parser.add_argument('--pde',
                    default=2, type=int,
                    help='id of the PDE model, default is 2')

parser.add_argument('--init_mesh',
                    default='edgemesh', type=str,
                    help='Type of mesh, default is EdgeMesh')

parser.add_argument('--space_degree',
        default=1, type=int,
        help='Degree of Finite Element Space, default is 1')

parser.add_argument('--beam_E',
                    default=2.07e11, type=float,
                    help='Type of mesh, default is the beam Young modulus')

parser.add_argument('--beam_nu',
                    default=0.276, type=float,
                    help='Type of mesh, default is the beam Poisson ratio')

parser.add_argument('--axle_E',
                    default=1.976e6, type=float,
                    help='Type of mesh, default is the axle Young modulus')

parser.add_argument('--axle_nu',
                    default=-0.5, type=float,
                    help='Type of mesh, default is the axle Poisson ratio')

parser.add_argument('--pbar_log',
                    default=True, type=bool,
                    help='Whether to show progress bar, default is True')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')

options = vars(parser.parse_args()) 

from fealpy.backend import backend_manager as bm
bm.set_backend(options['backend'])

from fealpy.csm.fem import TimobeamAxleModel
model = TimobeamAxleModel(options)
model.__str__()
model.timo_axle_system()
# K, F = model.apply_bc_penalty(K, F)
# K_dense = K.toarray()
# print('K', K_dense.shape)
# u_data = model.solve()
# print("u", u_data.reshape(-1, 6))


import pandas as pd
import numpy as np
file_path = r"C:\Users\Administrator\Desktop\K.xlsx"
df = pd.read_excel(file_path, header=None)
k_data = df.to_numpy()
# print("数组形状:", k_data.shape)
# print(np.array_equal(K_dense, k_data))   # 判断是否完全一样
# print(np.allclose(K_dense, k_data, rtol=1e-8, atol=1e-12))  # 判断数值上是否近似相等

# mask = ~np.isclose(K_dense, k_data, rtol=1e-8, atol=1e-12)
# rows, cols = np.where(mask)
# for r, c in zip(rows, cols):
#     print(f"不同元素在[{r}, {c}]: K_dense={K_dense[r,c]} k_data={k_data[r,c]}")
