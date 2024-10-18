"""
此脚本用于生成求解区域的边界 LB 算子的特征值和特征向量，并存储为单个 .npz 文件。
"""

from typing import Tuple
import argparse

import yaml
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import (
    BilinearForm,
    ScalarDiffusionIntegrator,
    ScalarMassIntegrator
)
from fealpy.mesh import IntervalMesh
from fealpy.mesh import TriangleMesh


parser = argparse.ArgumentParser()
parser.add_argument("config", type=str, help="config file")
parser.add_argument("--plot", help="plot the eigenvalues", action="store_true")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

Q_ = config['fem']['integral']
assert isinstance(Q_, int)
assert Q_ >= 1
P_ = config['fem']['order']


def laplace_eigen_fem(mesh: IntervalMesh, p: int = 1) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """_summary_

    Args:
        mesh (_type_): _description_
        p (int, optional): _description_. Defaults to 1.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: _description_
    """
    space = LagrangeFESpace(mesh, p=p)
    bform_0 = BilinearForm(space)
    bform_0.add_integrator(ScalarDiffusionIntegrator(q=Q_))
    A = bform_0.assembly().to_dense()
    bform_1 = BilinearForm(space)
    bform_1.add_integrator(ScalarMassIntegrator(q=Q_))
    M = bform_1.assembly().to_dense()
    w, v = eigh(A, M) # (gdof,) (gdof, gdof)

    return w, v, A, M

EXTx, EXTy = config['mesh']["ext"]
Lx, Ly = config['mesh']['length']
Origin = config['mesh']['origin']
refine = config['mesh']['refine']
box = [Origin[0], Origin[0] + Lx, Origin[1], Origin[1] + Ly]

print("Generating Boundary Laplace Beltrami operator...")
print(f"Config:")
print(f"  - Domain: {box[0:2]}x{box[2:4]}")
print(f"  - Integral points: {Q_}")
print(f"will be saved to file: {config['file']}", end='\n\n')
signal_ = input("Continue? (y/n) ")


if signal_ in {'y', 'Y'}:

    bm.set_backend('numpy')
    tri_mesh = TriangleMesh.from_box(box, EXTx, EXTy)
    mesh = IntervalMesh.from_mesh_boundary(tri_mesh)
    del tri_mesh

    bform_2 = BilinearForm(LagrangeFESpace(mesh))
    bform_2.add_integrator(ScalarMassIntegrator(q=Q_))
    M0 = bform_2.assembly().to_dense()
    del bform_2

    NN = mesh.number_of_nodes()
    mesh.uniform_refine(refine)

    w, v, _, _ = laplace_eigen_fem(mesh, p=P_)
    w = w[1:NN+1]
    vinv = v.T[1:NN+1, :NN] @ M0
    v = v[:NN, 1:NN+1]
    print(vinv @ v)

    np.savez(config['file'], w=w, v=v, vinv=vinv)
    print("Saved.")

    if args.plot:
        from matplotlib import pyplot as plt

        fig = plt.figure()

        axes = fig.add_subplot(111)
        f = np.arange(0, w.shape[0])
        axes.plot(f, np.sqrt(w))
        axes.plot(f, (f+1)/2 * np.pi/4)

        plt.show()

else:
    print("Aborted.")
